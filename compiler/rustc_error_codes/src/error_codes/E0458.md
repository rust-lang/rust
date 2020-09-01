An unknown "kind" was specified for a link attribute.

Erroneous code example:

```compile_fail,E0458
#[link(kind = "wonderful_unicorn")] extern "C" {}
// error: unknown kind: `wonderful_unicorn`
```

Please specify a valid "kind" value, from one of the following:

* static
* dylib
* framework
