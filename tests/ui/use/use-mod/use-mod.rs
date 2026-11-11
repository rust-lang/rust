use foo::bar::{
    self,
    Bar,
    self
//~^ ERROR the name `bar` is defined multiple times
};

use {self};
//~^ ERROR imports need to be explicitly named

mod foo {
    pub mod bar {
        pub struct Bar;
        pub struct Baz;
    }
}

fn main() {}
