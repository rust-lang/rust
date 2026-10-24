Attribute contains same meta item more than once.

Erroneous code example:

```compile_fail,E0538
#[deprecated(
    since="1.0.0",
    note="First deprecation note.",
    note="Second deprecation note." // error: multiple same meta item
)]
fn deprecated_function() {}
```

Meta items are the key-value pairs inside of an attribute. Each key may only be
used once in each attribute.

To fix the problem, remove all but one of the meta items with the same key.

Example:

```
#[deprecated(
    since="1.0.0",
    note="First deprecation note."
)]
fn deprecated_function() {}
```
