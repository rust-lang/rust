An unknown predicate was used inside the `cfg` attribute.

Erroneous code example:

```compile_fail,E0537
#[cfg(unknown())] // error: invalid predicate `unknown`
pub fn something() {}

pub fn main() {}
```

The `cfg` attribute supports only three kinds of predicates:

 * any
 * all
 * not

Example:

```
#[cfg(not(target_os = "linux"))] // ok!
pub fn something() {}

pub fn main() {}
```

For more information about the `cfg` attribute, read the section on
[Conditional Compilation][conditional-compilation] in the Reference.

[conditional-compilation]: https://doc.rust-lang.org/reference/conditional-compilation.html
