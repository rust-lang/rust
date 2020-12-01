A file wasn't found for an out-of-line module.

Erroneous code example:

```compile_fail,E0583
mod file_that_doesnt_exist; // error: file not found for module

fn main() {}
```

Please be sure that a file corresponding to the module exists. If you
want to use a module named `file_that_doesnt_exist`, you need to have a file
named `file_that_doesnt_exist.rs` or `file_that_doesnt_exist/mod.rs` in the
same directory.
