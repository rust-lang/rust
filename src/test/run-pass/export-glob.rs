// xfail-test
// Export the enum variants, without the enum

mod foo {
    export bar::*;
    mod bar {
        const a : int = 10;
    }
}

fn main() { let v = foo::a; }
