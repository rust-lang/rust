// Test that a glob-export functions as an explicit
// named export when referenced from outside its scope.

// Modified to not use export since it's going away. --pcw

mod foo {
    import bar::*;
    export a;
    mod bar {
        const a : int = 10;
    }
}

fn main() { let v = foo::a; }
