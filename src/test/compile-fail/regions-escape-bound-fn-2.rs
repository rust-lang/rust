fn with_int(f: fn(x: &int)) {
    let x = 3;
    f(&x);
}

fn main() {
    let mut x = none;
         //~^ ERROR reference is not valid outside of its lifetime
    with_int(|y| x = some(y));
}
