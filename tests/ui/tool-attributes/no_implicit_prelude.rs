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

    #[my_tool::attr] // Unaffected by the `no_implicit_prelude` attribute
    #[rustfmt::skip] // Unaffected by the `no_implicit_prelude` attribute
    #[clippy::format_args(Debug)]
    //~^ ERROR attribute macro `clippy::format_args` is ambiguous
    struct Foo;
}

fn main() {}
