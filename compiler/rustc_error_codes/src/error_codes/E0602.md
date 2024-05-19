An unknown or invalid lint was used on the command line.

Erroneous code example:

```sh
rustc -D bogus rust_file.rs
```

Maybe you just misspelled the lint name or the lint doesn't exist anymore.
Either way, try to update/remove it in order to fix the error.
