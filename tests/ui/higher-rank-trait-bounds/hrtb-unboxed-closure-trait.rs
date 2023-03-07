// run-pass
// Test HRTB used with the `Fn` trait.

fn foo<F:Fn(&isize)>(f: F) {
    let x = 22;
    f(&x);
}

fn main() {
    foo(|x: &isize| println!("{}", *x));
}
