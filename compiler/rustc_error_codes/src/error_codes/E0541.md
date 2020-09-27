An unknown meta item was used.

Erroneous code example:

```compile_fail,E0541
#[deprecated(
    since="1.0.0",
    // error: unknown meta item
    reason="Example invalid meta item. Should be 'note'")
]
fn deprecated_function() {}
```

Meta items are the key-value pairs inside of an attribute. The keys provided
must be one of the valid keys for the specified attribute.

To fix the problem, either remove the unknown meta item, or rename it if you
provided the wrong name.

In the erroneous code example above, the wrong name was provided, so changing
to a correct one it will fix the error. Example:

```
#[deprecated(
    since="1.0.0",
    note="This is a valid meta item for the deprecated attribute."
)]
fn deprecated_function() {}
```
