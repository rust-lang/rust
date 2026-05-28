// Regression test for #129021.

static UNIT: &'static &'static () = &&();

fn foo<'a, 'b, T>(_: &'a &'b (), v: &'b T) -> &'a T { v }

fn bad<'a, T>(x: &'a T) -> &'static T {
    let f: fn(_, &'a T) -> &'static T = foo;
    //~^ ERROR lifetime may not live long enough
    f(UNIT, x)
}

fn main() {}
