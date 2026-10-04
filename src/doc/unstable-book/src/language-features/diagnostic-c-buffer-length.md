# `diagnostic_c_buffer_length`

The tracking issue for this feature is: [#162563](https://github.com/rust-lang/rust/issues/162563)

------------------------

The `diagnostic_c_buffer_length` feature allows the
`#[diagnostic::c_buffer_length(pointer, length)]` attribute on functions and
foreign function declarations. The two arguments name distinct parameters of
the function: a raw pointer to a buffer and an integer length.

```rust
#![feature(diagnostic_c_buffer_length)]

unsafe extern "C" {
    #[diagnostic::c_buffer_length(buffer, length)]
    pub fn write(fd: i32, buffer: *const u8, length: usize) -> isize;
}
```

This is for a future lint described in [#148664](https://github.com/rust-lang/rust/issues/148664).

<!-- FIXME(JohnTitor): descirbe more once a lint is implemented. -->
