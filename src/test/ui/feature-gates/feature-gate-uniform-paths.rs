// edition:2018

pub mod foo {
    pub use bar::Bar; //~ ERROR imports can only refer to extern crate names

    pub mod bar {
        pub struct Bar;
    }
}

use inline; //~ ERROR imports can only refer to extern crate names

use Vec; //~ ERROR imports can only refer to extern crate names

use vec; //~ ERROR imports can only refer to extern crate names

fn main() {
    let _ = foo::Bar;
}
