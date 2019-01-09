// Tests that callees correctly infer an ordering between free regions
// that appear in their parameter list.  See also
// regions-free-region-ordering-caller.rs

fn ordering1<'a, 'b>(x: &'a &'b usize) -> &'a usize {
    // It is safe to assume that 'a <= 'b due to the type of x
    let y: &'b usize = &**x;
    return y;
}

fn ordering2<'a, 'b>(x: &'a &'b usize, y: &'a usize) -> &'b usize {
    // However, it is not safe to assume that 'b <= 'a
    &*y //~ ERROR lifetime mismatch [E0623]
}

fn ordering3<'a, 'b>(x: &'a usize, y: &'b usize) -> &'a &'b usize {
    // Do not infer an ordering from the return value.
    let z: &'b usize = &*x;
    //~^ ERROR lifetime mismatch [E0623]
    panic!();
}

// see regions-free-region-ordering-callee-4.rs

fn ordering5<'a, 'b>(a: &'a usize, b: &'b usize, x: Option<&'a &'b usize>) {
    let z: Option<&'a &'b usize> = None;
}

fn main() {}
