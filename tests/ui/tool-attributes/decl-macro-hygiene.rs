#![feature(register_tool)]
#![feature(decl_macro)]
#![register_tool(my_tool)]

mod my_tool {
    pub use std::prelude::v1::test as x;
}

pub macro bar() {
    #[my_tool::x]
    //~^ ERROR attribute macro `my_tool::x` is ambiguous
    fn bar() {}
}

bar!();

fn main() {}
