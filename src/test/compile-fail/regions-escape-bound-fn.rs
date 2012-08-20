fn with_int(f: fn(x: &int)) {
    let x = 3;
    f(&x);
}

fn main() {
    let mut x: Option<&int> = None; //~ ERROR cannot infer
    with_int(|y| x = Some(y));
}
