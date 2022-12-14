# `raw_dylib`

The tracking issue for this feature is: [#58713]

[#58713]: https://github.com/rust-lang/rust/issues/58713

------------------------

The `raw_dylib` feature allows you to link against the implementations of functions in an `extern`
block without, on Windows, linking against an import library.

```rust,ignore (partial-example)
#![feature(raw_dylib)]

#[link(name="library", kind="raw-dylib")]
extern {
    fn extern_function(x: i32);
}

fn main() {
    unsafe {
        extern_function(14);
    }
}
```

## Limitations

This feature is unstable for the `x86` architecture, and stable for all other architectures.

This feature is only supported on Windows.

On the `x86` architecture, this feature supports only the `cdecl`, `stdcall`, `system`, `fastcall`, and
`vectorcall` calling conventions.
