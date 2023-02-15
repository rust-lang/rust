fn foo(_: &()) {}

fn main() {
    let x: for<'a> fn(&'a ()) = foo;
    let y: for<'a> fn(&'a ()) = foo;
    x == y; //~ ERROR: `==` cannot be applied

    let x: for<'a> fn(&'a ()) = foo;
    let y: fn(&()) = foo;
    x == y; //~ ERROR: `==` cannot be applied

    let x: fn(&()) = foo;
    let y: for<'a> fn(&'a ()) = foo;
    x == y; //~ ERROR: `==` cannot be applied

    let x: for<'a> fn(&'a ()) = foo;
    let y: for<'a> fn(&'a ()) = foo;
    x == foo; //~ ERROR: `==` cannot be applied
    y == foo; //~ ERROR: `==` cannot be applied
    foo == x; //~ ERROR: `==` cannot be applied
    //~^ ERROR mismatched types
    foo == y; //~ ERROR: `==` cannot be applied
    //~^ ERROR mismatched types

    let x: for<'a> fn(&'a ()) = foo;
    let y: fn(&'static ()) = foo;
    x == y; //~ ERROR: `==` cannot be applied

    let x: fn(&'static ()) = foo;
    let y: fn(&'static ()) = foo;
    foo == x; //~ ERROR: `==` cannot be applied
    //~^ ERROR mismatched types
}
