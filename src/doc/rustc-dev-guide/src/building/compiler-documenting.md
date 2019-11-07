# Documenting rustc

You might want to build documentation of the various components
available like the standard library. There’s two ways to go about this.
You can run rustdoc directly on the file to make sure the HTML is
correct, which is fast. Alternatively, you can build the documentation
as part of the build process through x.py. Both are viable methods
since documentation is more about the content.

## Document everything

```bash
./x.py doc
```

## If you want to avoid the whole Stage 2 build

```bash
./x.py doc --stage 1
```

First the compiler and rustdoc get built to make sure everything is okay
and then it documents the files.

## Document specific components

```bash
./x.py doc src/doc/book
./x.py doc src/doc/nomicon
./x.py doc src/doc/book src/libstd
```

Much like individual tests or building certain components you can build only
 the documentation you want.

## Document internal rustc items

Compiler documentation is not built by default. To enable it, modify config.toml:

```toml
[build]
compiler-docs = true
```

Note that when enabled,
documentation for internal compiler items will also be built.

### Compiler Documentation

The documentation for the rust components are found at [rustc doc].

[rustc doc]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/
