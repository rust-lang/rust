fn with_int(f: fn(x: &int)) {
    let x = 3;
    f(&x);
}

fn main() {
    let mut x = None;
         //~^ ERROR reference is not valid outside of its lifetime
    with_int(|y| x = Some(y));
}
