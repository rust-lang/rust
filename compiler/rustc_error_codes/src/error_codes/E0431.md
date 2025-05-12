An invalid `self` import was made.

Erroneous code example:

```compile_fail,E0431
use {self}; // error: `self` import can only appear in an import list with a
            //        non-empty prefix
```

You cannot import the current module into itself, please remove this import
or verify you didn't misspell it.
