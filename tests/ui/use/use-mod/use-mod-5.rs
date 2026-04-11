//@ check-pass

mod foo {
    pub mod bar {
        pub fn drop() {}
    }
}

use foo::bar::self;

fn main() {
    bar::drop();
}
