There is no file shebang for python that works on all platforms (#71818).
To minimize breakage we have chosen to make this work when `bash` is available, or when using python/python2/python3 directly.
Unfortunately, this breaks users using the `py` wrapper on Windows, where `bash` isn't supported outside of MingW shells.
Existing versions of `py` will see the bash shebang line and try to respect it and interpret the file with `bash`.

You can do one of the following things to get x.py working:
1. Use any of `python`, `python2`, `python3`, `py -2`, or `py -3` to invoke x.py.
2. Use a MingW shell (often installed as "Git Bash", or through Cygwin).
3. Set the default file handler for .py files, which will allow using `./x.py` directly: `ftype Python.File="C:\Windows\py.exe" "-3" "%L" %*`.
   `py` may be installed in a different location; use `where py` to determine where.
4. Set a default python interpreter for the `py` wrapper: Add
```ini
[commands]
bash=python
```
   to `%APPDATA%\py.ini`.

5. Wait until October and update to the latest `py` wrapper, which fixes this bug: https://github.com/python/cpython/issues/94399#issuecomment-1170649407
