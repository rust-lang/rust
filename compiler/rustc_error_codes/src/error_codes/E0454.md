A link name was given with an empty name.

Erroneous code example:

```compile_fail,E0454
#[link(name = "")] extern "C" {}
// error: `#[link(name = "")]` given with empty name
```

The rust compiler cannot link to an external library if you don't give it its
name. Example:

```no_run
#[link(name = "some_lib")] extern "C" {} // ok!
```
