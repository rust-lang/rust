// xfail-test

// Test that a glob-export functions as an explicit
// named export when referenced from outside its scope.

mod foo {
    export bar::*;
    mod bar {
        const a : int = 10;
    }
}

fn main() { let v = foo::a; }
