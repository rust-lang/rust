# `register_tool`

The tracking issue for this feature is: [#66079]

[#66079]: https://github.com/rust-lang/rust/issues/66079

------------------------

The `register_tool` language feature informs the compiler that attributes in your code are meant to be used with tools other than the compiler itself. This can be useful if your code has semantic meaning without the external tool, but enables additional features when the tool is present.

`register_tool` also allows configuring lint levels for external tools.

Tool attributes are only meant for ignorable attributes. If your code *changes* meaning when the attribute is present, it should not use a tool attribute (because it cannot be compiled with anything other than the external tool, and in a sense is a fork of the language).

------------------------

`#![register_tool(tool)]` is an attribute, and is only valid at the crate root.
Attributes using the registered tool are checked for valid syntax, and lint attributes are checked to be in a valid format. However, the compiler cannot validate the semantics of the attribute, nor can it tell whether the configured lint is present in the external tool.

Semantically, `clippy::*`, `rustdoc::*`, and `rustfmt::*` lints and attributes all behave as if `#![register_tool(clippy, rustdoc, rustfmt)]` were injected into the crate root, except that the `rustdoc` namespace can only be used for lints, not for attributes.
When compiling with `-Z unstable-features`, `rustc::*` lints can also be used. Like `rustdoc`, the `rustc` namespace can only be used with lints, not attributes.

The compiler will emit an error if it encounters a lint/attribute whose namespace isn't a registered tool.

Tool namespaces cannot be nested; `register_tool(main_tool::subtool)` is an error.

## Examples

Tool attributes:

```rust
#![feature(register_tool)]
#![register_tool(c2rust)]

// Mark which C header file this module was generated from.
#[c2rust::header_src = "operations.h"]
pub mod operations_h {
    use std::ffi::c_int;

    // Mark which source line this struct was generated from.
    #[c2rust::src_loc = "11:0"]
    pub struct Point {
        pub x: c_int,
        pub y: c_int,
    }
}
```

Tool lints:

```
#![feature(register_tool)]
#![register_tool(bevy)]
#![deny(bevy::duplicate_bevy_dependencies)]
```
