% Using traits to share implementations

> **[FIXME]** Elaborate.

> **[FIXME]** We probably want to discourage this, at least when used in a way
> that is publicly exposed.

Traits that provide default implementations for function can provide code reuse
across types. For example, a `print` method can be defined across multiple
types as follows:

``` Rust
trait Printable {
    // Default method implementation
    fn print(&self) { println!("{:?}", *self) }
}

impl Printable for i32 {}

impl Printable for String {
    fn print(&self) { println!("{}", *self) }
}

impl Printable for bool {}

impl Printable for f32 {}
```

This allows the implementation of `print` to be shared across types, yet
overridden where needed, as seen in the `impl` for `String`.
