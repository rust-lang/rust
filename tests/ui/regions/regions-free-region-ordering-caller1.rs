// Test various ways to construct a pointer with a longer lifetime
// than the thing it points at and ensure that they result in
// errors. See also regions-free-region-ordering-callee.rs

fn call1<'a>(x: &'a usize) {
    // Test that creating a pointer like
    // &'a &'z usize requires that 'a <= 'z:
    let y: usize = 3;
    let z: &'a & usize = &(&y);
    //~^ ERROR temporary value dropped while borrowed
    //~^^ ERROR `y` does not live long enough
}

fn main() {}
