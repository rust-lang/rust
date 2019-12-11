The `feature` attribute was badly formed.

Erroneous code example:

```compile_fail,E0556
#![feature(foo_bar_baz, foo(bar), foo = "baz", foo)] // error!
#![feature] // error!
#![feature = "foo"] // error!
```

The `feature` attribute only accept a "feature flag" and can only be used on
nightly. Example:

```ignore (only works in nightly)
#![feature(flag)]
```
