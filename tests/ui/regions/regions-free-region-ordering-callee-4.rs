// Tests that callees correctly infer an ordering between free regions
// that appear in their parameter list.  See also
// regions-free-region-ordering-caller.rs

fn ordering4<'a, 'b, F>(a: &'a usize, b: &'b usize, x: F) where F: FnOnce(&'a &'b usize) {
    //~^ ERROR reference has a longer lifetime than the data it references
    // Do not infer ordering from closure argument types.
    let z: Option<&'a &'b usize> = None;
}

fn main() {}
