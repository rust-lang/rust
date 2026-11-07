//@ edition:2021

#![feature(decl_macro)]

mod foo {
    macro_rules! bar {
        () => {};
    }

    pub use bar as _; //~ ERROR `bar` is only public within the crate, and cannot be re-exported outside

    macro baz() {}

    pub use baz as _; //~ ERROR `baz` is private, and cannot be re-exported
}

fn main() {}
