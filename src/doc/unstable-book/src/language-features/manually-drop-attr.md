# `manually_drop_attr`

The tracking issue for this feature is: [#100344]

[#100344]: https://github.com/rust-lang/rust/issues/100344

The `manually_drop_attr` feature enables the `#[manually_drop]` attribute, which disables the drop glue for the type it is applied to.

For example, `std::mem::ManuallyDrop` is implemented as follows:

```rs
#[manually_drop]
struct ManuallyDrop<T>(T);
```

But you can also use the attribute to change the order in which fields are dropped, by overriding `Drop`:

```rs
/// This struct changes the order in which `x` and `y` are dropped from the default.
#[manually_drop]
struct MyStruct {
    x: String,
    y: String,
}

impl Drop for MyStruct {
    fn drop(&mut self) {
        unsafe {
            std::ptr::drop_in_place(&mut self.y);
            std::ptr::drop_in_place(&mut self.x);
        }
    }
}
```

This can be useful in combination with `repr(C)`, to decouple the layout from the destruction order. See MCP [#135](https://github.com/rust-lang/lang-team/issues/135).
