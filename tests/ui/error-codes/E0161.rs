// Check that E0161 is a hard error in all possible configurations that might
// affect it.

//@ revisions: base ul
//@[base] check-fail
//@[ul] check-pass

#![allow(incomplete_features)]
#![cfg_attr(ul, feature(unsized_locals))]

trait Bar {
    fn f(self);
}

fn foo(x: Box<dyn Bar>) {
    x.f();
    //[base]~^ ERROR E0161
}

fn main() {}
