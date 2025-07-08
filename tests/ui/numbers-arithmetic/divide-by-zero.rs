//@ run-fail
//@ check-run-results
//@ needs-subprocess

#[allow(unconditional_panic)]
fn main() {
    let y = 0;
    let _z = 1 / y;
}
