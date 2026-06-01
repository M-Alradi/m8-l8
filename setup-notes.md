# Setup Notes

Record any setup issues (and their resolutions) here as you encounter them.
This file is for your own troubleshooting record and for the SI/TA team to
reference if they need to help debug.

---
## Issue 1 — Windows UnicodeDecodeError during pytest corpus loading

**Problem:**
Running tests on Windows fails with:

```
UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d

```

Specifically in:

```
tests/test_retrieval.py::test_index_corpus_ingests_full_corpus

```

**Cause:**
The test suite opens data files without explicitly defining an encoding profile. On Windows, Python falls back to the system locale default (`cp1252` / `charmap`), which crashes when parsing UTF-8 technical text or special formatting symbols present in the CQADupStack dataset.

**Impact:**

* Running `pytest` fails locally on Windows development machines.
* CI/CD pipelines or Linux/macOS environments do not replicate the issue since they default natively to UTF-8.

**Workaround / Fix:**
Because the autograder test code cannot be altered locally, the encoding must be forced globally at the environment level.

In the Bash CLI (Git Bash), export the Python UTF-8 environment variable prior to executing the test framework:

```bash
export PYTHONUTF8=1
pytest tests/

```

To make this change permanent across all terminal sessions, append `export PYTHONUTF8=1` to the local `~/.bashrc` configuration.