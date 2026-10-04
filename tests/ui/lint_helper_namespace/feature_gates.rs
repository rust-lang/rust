#![crate_type = "lib"]
#![feature(register_tool)]
#![register_tool(lint)]

struct X;

impl X {
    #[lint::as_ptr] //~ ERROR use of an internal attribute
    #[lint::never_returns_null_ptr] //~ ERROR use of an internal attribute
    #[lint::should_not_be_called_on_const_items] //~ ERROR use of an internal attribute
    fn cast(x: &u8) -> *const u8 {
        x
    }
}
