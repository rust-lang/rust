//@ run-pass
#![allow(unreachable_code)]
fn main() {
    return ();

    let x = ();
    //~^ WARN unused variable: `x`
    x
}
