//@ run-rustfix

// Point at the captured immutable outer variable

#![feature(ergonomic_clones)]
#![allow(incomplete_features)]

fn foo(mut f: Box<dyn FnMut()>) {
    f();
}

fn main() {
    let y = true;
    foo(Box::new(use || y = !y) as Box<_>);
    //~^ ERROR cannot assign to `y`, as it is not declared as mutable
}
