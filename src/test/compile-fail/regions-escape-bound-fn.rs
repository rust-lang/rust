fn with_int(f: fn(x: &int)) {
    let x = 3;
    f(&x);
}

fn main() {
    let mut x: option<&int> = none; //~ ERROR cannot infer
    with_int(|y| x = some(y));
}
