// tests that the anonymous_parameters lint is warn-by-default on the 2018 edition

// compile-pass
// edition:2018
// run-rustfix

trait Foo {
    fn foo(u8);
    //^ WARN anonymous parameters are deprecated
    //| WARN this was previously accepted
}

fn main() {}
