// run-rustfix

// Point at the captured immutable outer variable

fn foo(mut f: Box<dyn FnMut()>) {
    f();
}

fn main() {
    let y = true;
    foo(Box::new(move || y = false) as Box<_>);
    //~^ ERROR cannot assign to `y`, as it is not declared as mutable
}
