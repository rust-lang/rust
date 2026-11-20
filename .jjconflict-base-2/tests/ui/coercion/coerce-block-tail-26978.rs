//@ check-fail
fn f(_: &i32) {}

fn main() {
    let x = Box::new(1i32);

    f(&x);
    f(&(x));
    f(&{x});
    //~^ ERROR mismatched types
}
