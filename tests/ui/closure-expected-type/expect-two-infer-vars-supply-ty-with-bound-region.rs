fn with_closure<F, A, B>(_: F)
    where F: FnOnce(A, B)
{
}

fn a() {
    // Type of `y` is unconstrained.
    with_closure(|x: u32, y| {}); //~ ERROR E0282
}

fn b() {
    with_closure(|x: u32, y: u32| {}); // OK
}

fn c() {
    with_closure(|x: u32, y: u32| {}); // OK
}

fn main() { }
