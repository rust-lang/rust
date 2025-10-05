// Issue #25860: This exploit is now FIXED and should fail to compile
// Previously marked as known-bug, now correctly rejected

static UNIT: &'static &'static () = &&();

fn foo<'a, 'b, T>(_: &'a &'b (), v: &'b T, _: &()) -> &'a T { v }

fn bad<'a, T>(x: &'a T) -> &'static T {
    let f: fn(_, &'a T, &()) -> &'static T = foo;
    //~^ ERROR mismatched types
    f(UNIT, x, &())
}

fn main() {}
