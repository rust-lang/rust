This error indicates that a `#[non_exhaustive]` attribute was incorrectly placed
on something other than a struct or enum.

Erroneous code example:

```compile_fail,E0701
#[non_exhaustive]
trait Foo { }
```
