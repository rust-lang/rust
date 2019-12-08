An invalid meta-item was used inside an attribute.

Erroneous code example:

```compile_fail,E0551
#[deprecated(note)] // error!
fn i_am_deprecated() {}
```

Meta items are the key-value pairs inside of an attribute. To fix this issue,
you need to give a value to the `note` key. Example:

```
#[deprecated(note = "because")] // ok!
fn i_am_deprecated() {}
```
