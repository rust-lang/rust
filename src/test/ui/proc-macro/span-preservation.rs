//~ ERROR mismatched types
// aux-build:span-preservation.rs

// For each of these, we should get the appropriate type mismatch error message,
// and the function should be echoed.

extern crate span_preservation as foo;

use foo::foo;

#[foo]
fn a() {
    let x: usize = "hello";;;;; //~ ERROR mismatched types
}

#[foo]
fn b(x: Option<isize>) -> usize {
    match x {
        Some(x) => { return x }, //~ ERROR mismatched types
        None => 10
    }
}

#[foo]
fn c() {
    struct Foo {
        a: usize
    }

    struct Bar {
        a: usize,
        b: usize
    }

    let x = Foo { a: 10isize }; //~ ERROR mismatched types
    let y = Foo { a: 10, b: 10isize }; //~ ERROR has no field named `b`
}

// FIXME: This doesn't work at the moment. See the one below. The pretty-printer
// injects a "C" between `extern` and `fn` which causes a "probably_eq"
// `TokenStream` mismatch. The lack of `"C"` should be preserved in the AST.
#[foo]
extern fn bar() {
    0
}

#[foo]
extern "C" fn baz() {
    0 //~ ERROR mismatched types
}

fn main() {}
