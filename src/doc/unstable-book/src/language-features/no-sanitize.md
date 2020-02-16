# `no_sanitize`

The tracking issue for this feature is: [#39699]

[#39699]: https://github.com/rust-lang/rust/issues/39699

------------------------

The `no_sanitize` attribute can be used to selectively disable sanitizer
instrumentation in an annotated function. This might be useful to: avoid
instrumentation overhead in a performance critical function, or avoid
instrumenting code that contains constructs unsupported by given sanitizer.

The precise effect of this annotation depends on particular sanitizer in use.
For example, with `no_sanitize(thread)`, the thread sanitizer will no longer
instrument non-atomic store / load operations, but it will instrument atomic
operations to avoid reporting false positives and provide meaning full stack
traces.

## Examples

``` rust
#![feature(no_sanitize)]

#[no_sanitize(address)]
fn foo() {
  // ...
}
```
