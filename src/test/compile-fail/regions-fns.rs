// Before fn subtyping was properly implemented,
// we reported errors in this case:

fn not_ok(a: &uint, b: &b/uint) {
    let mut g: fn@(x: &uint) = fn@(x: &b/uint) {};
    //~^ ERROR mismatched types
    g(a);
}

fn main() {
}
