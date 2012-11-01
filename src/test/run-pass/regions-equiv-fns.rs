// Before fn subtyping was properly implemented,
// we reported errors in this case:

fn ok(a: &uint) {
    // Here &r is an alias for &:
    let mut g: fn@(x: &uint) = fn@(x: &r/uint) {};
    g(a);
}

fn main() {
}


