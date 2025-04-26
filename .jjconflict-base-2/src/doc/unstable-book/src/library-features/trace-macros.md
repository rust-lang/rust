# `trace_macros`

The tracking issue for this feature is [#29598].

[#29598]: https://github.com/rust-lang/rust/issues/29598

------------------------

With `trace_macros` you can trace the expansion of macros in your code.

## Examples

```rust
#![feature(trace_macros)]

fn main() {
    trace_macros!(true);
    println!("Hello, Rust!");
    trace_macros!(false);
}
```

The `cargo build` output:

```txt
note: trace_macro
 --> src/main.rs:5:5
  |
5 |     println!("Hello, Rust!");
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^
  |
  = note: expanding `println! { "Hello, Rust!" }`
  = note: to `print ! ( concat ! ( "Hello, Rust!" , "\n" ) )`
  = note: expanding `print! { concat ! ( "Hello, Rust!" , "\n" ) }`
  = note: to `$crate :: io :: _print ( format_args ! ( concat ! ( "Hello, Rust!" , "\n" ) )
          )`

    Finished dev [unoptimized + debuginfo] target(s) in 0.60 secs
```
