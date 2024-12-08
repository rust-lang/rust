A link was used without a name parameter.

Erroneous code example:

```compile_fail,E0459
#[link(kind = "dylib")] extern "C" {}
// error: `#[link(...)]` specified without `name = "foo"`
```

Please add the name parameter to allow the rust compiler to find the library
you want. Example:

```no_run
#[link(kind = "dylib", name = "some_lib")] extern "C" {} // ok!
```
