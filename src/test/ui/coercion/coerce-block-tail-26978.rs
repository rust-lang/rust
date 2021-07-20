// build-pass
fn f(_: &i32) {}

fn main() {
    let x = Box::new(1i32);

    f(&x); // OK
    f(&(x)); // OK
    f(&{x}); // Error
}
