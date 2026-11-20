A metadata file was invalid.

Erroneous code example:

```ignore (needs extern files)
use ::foo; // error: found invalid metadata files for crate `foo`
```

When loading crates, each crate must have a valid metadata file.
Invalid files could be caused by filesystem corruption,
an IO error while reading the file, or (rarely) a bug in the compiler itself.

Consider deleting the file and recreating it,
or reporting a bug against the compiler.
