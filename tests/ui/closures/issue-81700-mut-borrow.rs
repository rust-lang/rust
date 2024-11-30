//@ check-pass
fn foo(x: &mut u32) {
    let bar = || { foo(x); };
    bar(); //~ WARNING cannot borrow
}
fn main() {}
