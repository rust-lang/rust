The `#![feature]` attribute specified an unknown feature.

Erroneous code example:

```compile_fail,E0635
#![feature(nonexistent_rust_feature)] // error: unknown feature
```
