#![crate_type = "lib"]
#![feature(register_tool)]
#![register_tool(lint)]

#[lint::as_ptr] //~ ERROR use of an internal attribute
#[lint::never_returns_null_ptr] //~ ERROR use of an internal attribute
fn cast(x: &u8) -> *const u8 {
    x
}
