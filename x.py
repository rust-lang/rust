#!/usr/bin/env python3
# Some systems don't have `python3` in their PATH. This isn't supported by x.py directly;
# they should use `x` or `x.ps1` instead.

# This file is only a "symlink" to bootstrap.py — all logic should go there.

# Parts of `bootstrap.py` use the `multiprocessing` module, so this entry point
# must use the normal `if __name__ == '__main__':` convention to avoid problems.

if __name__ == "__main__":
    import os
    import sys
    import warnings
    from inspect import cleandoc

    major = sys.version_info.major
    minor = sys.version_info.minor

    # If this is Python 2, try to re-execute using Python 3.
    if major < 3:
        try:
            os.execvp("py", ["py", "-3"] + sys.argv)
        except OSError:
            try:
                os.execvp("python3", ["python3"] + sys.argv)
            except OSError:
                sys.exit(
                    "Error: Python 3 is required to run this script, "
                    "but it was not found on your system."
                )

    # Soft deprecation of old Python versions (< 3.8)
    skip_check = os.environ.get("RUST_IGNORE_OLD_PYTHON") == "1"
    if not skip_check and (major < 3 or (major == 3 and minor < 8)):
        msg = cleandoc(
            f"""
            Using Python {major}.{minor}, but >= 3.8 is recommended.
            Your Python version should continue to work for now,
            but this may change in the future. If Python >= 3.8 is
            not available on your system, please file an issue to
            help us understand timelines.

            This message can be suppressed by setting:
            RUST_IGNORE_OLD_PYTHON=1
            """
        )
        warnings.warn(msg, stacklevel=2)

    rust_dir = os.path.dirname(os.path.abspath(__file__))
    bootstrap_path = os.path.join(rust_dir, "src", "bootstrap")

    # Verify that the bootstrap path exists
    if not os.path.isdir(bootstrap_path):
        sys.exit(f"Error: Expected bootstrap directory not found at: {bootstrap_path}")

    # Add bootstrap path to module search path
    sys.path.insert(0, bootstrap_path)

    try:
        import bootstrap
    except ImportError as e:
        sys.exit(f"Error importing bootstrap module: {e}")

    try:
        bootstrap.main()
    except Exception as e:
        sys.exit(f"Bootstrap failed: {e}")
