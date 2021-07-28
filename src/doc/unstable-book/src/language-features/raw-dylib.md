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

Currently, this feature is only supported on `-windows-msvc` targets.  Non-Windows platforms don't have import
libraries, and an incompatibility between LLVM and the BFD linker means that it is not currently supported on
`-windows-gnu` targets.

On the `i686-pc-windows-msvc` target, this feature supports only the `cdecl`, `stdcall`, `system`, and `fastcall`
calling conventions.
