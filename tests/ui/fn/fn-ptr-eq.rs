// check-pass

fn foo(_: &()) {}

fn main() {
    let x: for<'a> fn(&'a ()) = foo;
    let y: for<'a> fn(&'a ()) = foo;
    bar(x, y);

    let x: fn(&'static ()) = foo;
    let y: for<'a> fn(&'a ()) = foo;
    x == y;
    let x: fn(&'static ()) = foo;
    let y: fn(&'static ()) = foo;
    x == y;
    x == foo;
}

fn bar<T: PartialEq>(_: T, _: T) {}
