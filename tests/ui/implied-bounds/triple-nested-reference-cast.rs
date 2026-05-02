// Regression test: triple nested reference cast exploiting #25860
// Tests that multiple implied bounds ('b: 'a AND 'c: 'b) are all checked.

fn triple<'a, 'b, 'c, T>(_: &'a &'b &'c (), v: &'c T) -> &'a T { v }

fn exploit<'a, 'c, T>(x: &'a T) -> &'c T {
    let f: fn(_, &'a T) -> &'c T = triple;
    //~^ ERROR lifetime may not live long enough
    f(&&&&(), x)
}

fn main() {}
