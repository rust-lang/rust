//@ check-pass

mod foo {
    pub mod bar {
        pub fn drop() {}
    }
}

use foo::bar::self as abc;

fn main() {
    abc::drop();
}
