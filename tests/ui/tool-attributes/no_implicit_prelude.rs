#![no_implicit_prelude]
#![feature(register_tool)]
#![register_tool(my_tool)]

mod test1 {
    mod my_tool {
        pub use ::std::derive as attr;
    }

    #[my_tool::attr(Debug)]
    //~^ ERROR attribute macro `my_tool::attr` is ambiguous
    struct Foo;
}

mod test2 {
    mod clippy {
        pub use ::std::derive as format_args;
    }

    #[rustfmt::skip]
    //~^ ERROR cannot find module or crate `rustfmt` in this scope
    #[clippy::format_args(Debug)]
    //~^ ERROR attribute macro `clippy::format_args` is ambiguous
    struct Foo;
}

fn main() {}
