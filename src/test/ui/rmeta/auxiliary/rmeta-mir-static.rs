// no-prefer-dynamic
// compile-flags: --emit=metadata -Zalways-encode-mir=yes
#![crate_type="rlib"]

pub static FOO: &str = "foo";
pub const BAR: i32 = 123;
