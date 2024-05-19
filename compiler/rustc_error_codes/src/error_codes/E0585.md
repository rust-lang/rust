A documentation comment that doesn't document anything was found.

Erroneous code example:

```compile_fail,E0585
fn main() {
    // The following doc comment will fail:
    /// This is a useless doc comment!
}
```

Documentation comments need to be followed by items, including functions,
types, modules, etc. Examples:

```
/// I'm documenting the following struct:
struct Foo;

/// I'm documenting the following function:
fn foo() {}
```
