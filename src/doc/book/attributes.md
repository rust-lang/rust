% Attributes

Declarations can be annotated with ‘attributes’ in Rust. They look like this:

```rust
#[test]
# fn foo() {}
```

or like this:

```rust
# mod foo {
#![test]
# }
```

The difference between the two is the `!`, which changes what the attribute
applies to:

```rust,ignore
#[foo]
struct Foo;

mod bar {
    #![bar]
}
```

The `#[foo]` attribute applies to the next item, which is the `struct`
declaration. The `#![bar]` attribute applies to the item enclosing it, which is
the `mod` declaration. Otherwise, they’re the same. Both change the meaning of
the item they’re attached to somehow.

For example, consider a function like this:

```rust
#[test]
fn check() {
    assert_eq!(2, 1 + 1);
}
```

It is marked with `#[test]`. This means it’s special: when you run
[tests][tests], this function will execute. When you compile as usual, it won’t
even be included. This function is now a test function.

[tests]: testing.html

Attributes may also have additional data:

```rust
#[inline(always)]
fn super_fast_fn() {
# }
```

Or even keys and values:

```rust
#[cfg(target_os = "macos")]
mod macos_only {
# }
```

Rust attributes are used for a number of different things. There is a full list
of attributes [in the reference][reference]. Currently, you are not allowed to
create your own attributes, the Rust compiler defines them.

[reference]: ../reference.html#attributes
