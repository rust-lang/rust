An unknown argument was given to the `inline` attribute.

Erroneous code example:

```compile_fail,E0535
#[inline(unknown)] // error: invalid argument
pub fn something() {}

fn main() {}
```

The `inline` attribute only supports two arguments:

 * always
 * never

All other arguments given to the `inline` attribute will return this error.
Example:

```
#[inline(never)] // ok!
pub fn something() {}

fn main() {}
```

For more information see the [`inline` Attribute][inline-attribute] section
of the Reference.

[inline-attribute]: https://doc.rust-lang.org/reference/attributes/codegen.html#the-inline-attribute
