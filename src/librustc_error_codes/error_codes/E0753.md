An inner doc comment was used in an invalid context.

Erroneous code example:

```compile_fail,E0753
fn foo() {}
//! foo
// ^ error!
fn main() {}
```

Inner document can only be used before items. For example:

```
//! A working comment applied to the module!
fn foo() {
    //! Another working comment!
}
fn main() {}
```

In case you want to document the item following the doc comment, you might want
to use outer doc comment:

```
/// I am an outer doc comment
#[doc = "I am also an outer doc comment!"]
fn foo() {
    // ...
}
```
