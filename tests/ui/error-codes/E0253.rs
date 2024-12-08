mod foo {
    pub trait MyTrait {
        fn do_something();
    }
}

use foo::MyTrait::do_something;
    //~^ ERROR E0253

fn main() {}
