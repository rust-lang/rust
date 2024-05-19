// https://github.com/rust-lang/rust/pull/114682#discussion_r1420534109

#![feature(decl_macro)]

macro_rules! mac {
    () => {
        pub macro A() {
            println!("non import")
        }
    }
}

mod m {
    pub macro A() {
        println!("import")
    }
}

pub use m::*;
mac!();

fn main() {
    A!();
    //~^ ERROR `A` is ambiguous
}
