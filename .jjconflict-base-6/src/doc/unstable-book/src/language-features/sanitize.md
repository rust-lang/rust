# `sanitize`

The tracking issue for this feature is: [#39699]

[#39699]: https://github.com/rust-lang/rust/issues/39699

------------------------

The `sanitize` attribute can be used to selectively disable or enable sanitizer
instrumentation in an annotated function. This might be useful to: avoid
instrumentation overhead in a performance critical function, or avoid
instrumenting code that contains constructs unsupported by given sanitizer.

The precise effect of this annotation depends on particular sanitizer in use.
For example, with `sanitize(thread = "off")`, the thread sanitizer will no
longer instrument non-atomic store / load operations, but it will instrument
atomic operations to avoid reporting false positives and provide meaning full
stack traces.

This attribute was previously named `no_sanitize`.

## Examples

``` rust
#![feature(sanitize)]

#[sanitize(address = "off")]
fn foo() {
  // ...
}
```

It is also possible to disable sanitizers for entire modules and enable them
for single items or functions.

```rust
#![feature(sanitize)]

#[sanitize(address = "off")]
mod foo {
  fn unsanitized() {
    // ...
  }

  #[sanitize(address = "on")]
  fn sanitized() {
    // ...
  }
}
```

It's also applicable to impl blocks.

```rust
#![feature(sanitize)]

trait MyTrait {
  fn foo(&self);
  fn bar(&self);
}

#[sanitize(address = "off")]
impl MyTrait for () {
  fn foo(&self) {
    // ...
  }

  #[sanitize(address = "on")]
  fn bar(&self) {
    // ...
  }
}
```
