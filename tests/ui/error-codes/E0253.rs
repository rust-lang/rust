mod foo {
    pub trait MyTrait {
        type SomeType;
    }
}

use foo::MyTrait::SomeType;
    //~^ ERROR E0253

fn main() {}
