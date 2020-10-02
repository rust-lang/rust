An `export_name` attribute contains null characters (`\0`).

Erroneous code example:

```compile_fail,E0648
#[export_name="\0foo"] // error: `export_name` may not contain null characters
pub fn bar() {}
```

To fix this error, remove the null characters:

```
#[export_name="foo"] // ok!
pub fn bar() {}
```
