#![allow(dead_code, unused_variables)]

pub mod foo {
    #[derive(Default)]
    pub struct Foo { pub visible: bool, invisible: bool, }
}

fn main() {
    let foo::Foo {} = foo::Foo::default();
    //~^ ERROR pattern does not mention field `visible` and inaccessible fields
}
