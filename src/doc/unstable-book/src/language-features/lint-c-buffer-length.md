# `lint_c_buffer_length`

The tracking issue for this feature is: [#162563](https://github.com/rust-lang/rust/issues/162563)

------------------------

The `lint_c_buffer_length` feature allows the
`#[lint::c_buffer_length(pointer, length)]` attribute on functions and
foreign function declarations. The two arguments name distinct parameters of
the function: a raw pointer to a buffer and an integer length.

Using this attribute also requires enabling `register_tool` and registering the
`lint` tool namespace.

```rust
#![feature(lint_c_buffer_length, register_tool)]
#![register_tool(lint)]

unsafe extern "C" {
    #[lint::c_buffer_length(buffer, length)]
    pub fn write(fd: i32, buffer: *const u8, length: usize) -> isize;
}
```

This is for a future lint described in [#148664](https://github.com/rust-lang/rust/issues/148664).
The attribute currently validates the parameter names and types, but does not
check calls to the annotated function.

<!-- FIXME(JohnTitor): describe more once a lint is implemented. -->
