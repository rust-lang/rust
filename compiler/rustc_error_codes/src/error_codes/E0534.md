The `inline` attribute was malformed.

Erroneous code example:

```compile_fail,E0534
#[inline()] // error: expected one argument
pub fn something() {}

fn main() {}
```

The parenthesized `inline` attribute requires the parameter to be specified:

```
#[inline(always)]
fn something() {}
```

or:

```
#[inline(never)]
fn something() {}
```

Alternatively, a paren-less version of the attribute may be used to hint the
compiler about inlining opportunity:

```
#[inline]
fn something() {}
```

For more information see the [`inline` attribute][inline-attribute] section
of the Reference.

[inline-attribute]: https://doc.rust-lang.org/reference/attributes/codegen.html#the-inline-attribute
