// proc-macro: rustfmt.rs
#![feature(custom_inner_attributes)]
#![feature(macro_attr)]

mod a {
    #![clippy::ignore]
    //~^ ERROR cannot find `ignore` in `clippy`
    //~| ERROR `clippy` is ambiguous

    pub mod clippy {}
}

mod b {
    #![clippy::ignore]
    //~^ ERROR cannot find `ignore` in `clippy`
    //~| ERROR `clippy` is ambiguous

    use a::clippy;
}

fn main() {}
