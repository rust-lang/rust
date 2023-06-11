#!/usr/bin/env python3
# Some systems don't have `python3` in their PATH. This isn't supported by x.py directly;
# they should use `x` or `x.ps1` instead.

# This file is only a "symlink" to bootstrap.py, all logic should go there.

# Parts of `bootstrap.py` use the `multiprocessing` module, so this entry point
# must use the normal `if __name__ == '__main__':` convention to avoid problems.
if __name__ == '__main__':
    import os
    import sys
    import warnings
    from inspect import cleandoc

    major = sys.version_info.major
    minor = sys.version_info.minor

    # If this is python2, check if python3 is available and re-execute with that
    # interpreter. Only python3 allows downloading CI LLVM.
    #
    # This matters if someone's system `python` is python2.
    if major < 3:
        try:
            os.execvp("py", ["py", "-3"] + sys.argv)
        except OSError:
            try:
                os.execvp("python3", ["python3"] + sys.argv)
            except OSError:
                # Python 3 isn't available, fall back to python 2
                pass

    # soft deprecation of old python versions
    skip_check = os.environ.get("RUST_IGNORE_OLD_PYTHON") == "1"
    if major < 3 or (major == 3 and minor < 6):
        msg = cleandoc("""
            Using python {}.{} but >= 3.6 is recommended. Your python version
            should continue to work for the near future, but this will
            eventually change. If python >= 3.6 is not available on your system,
            please file an issue to help us understand timelines.

            This message can be suppressed by setting `RUST_IGNORE_OLD_PYTHON=1`
        """.format(major, minor))
        warnings.warn(msg)

    rust_dir = os.path.dirname(os.path.abspath(__file__))
    # For the import below, have Python search in src/bootstrap first.
    sys.path.insert(0, os.path.join(rust_dir, "src", "bootstrap"))

    import bootstrap
    bootstrap.main()
