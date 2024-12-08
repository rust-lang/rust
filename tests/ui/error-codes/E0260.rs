extern crate alloc;

mod alloc {
//~^ ERROR the name `alloc` is defined multiple times [E0260]
    pub trait MyTrait {
        fn do_something();
    }
}

fn main() {}
