#![feature(mut_restriction)]

pub mod foo {
    pub struct Foo {
        pub mut(self) alpha: u8,
    }

    pub enum Bar {
        Beta(mut(self) u8),
    }
}

fn main() {
    let foo = foo::Foo { alpha: 0 }; //~ ERROR
    let bar = foo::Bar::Beta(0); //~ ERROR
}
