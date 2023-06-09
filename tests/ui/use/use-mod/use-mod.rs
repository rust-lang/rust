use foo::bar::{
    self,
//~^ ERROR `self` import can only appear once in an import list
    Bar,
    self
//~^ ERROR the name `bar` is defined multiple times
};

use {self};
//~^ ERROR `self` import can only appear in an import list with a non-empty prefix

mod foo {
    pub mod bar {
        pub struct Bar;
        pub struct Baz;
    }
}

fn main() {}
