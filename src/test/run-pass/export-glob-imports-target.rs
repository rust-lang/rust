// Test that a glob-export functions as an import
// when referenced within its own local scope.

mod foo {
    export bar::*;
    mod bar {
        const a : int = 10;
    }
    fn zum() {
        let b = a;
    }
}

fn main() { }
