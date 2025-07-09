//@ run-fail
//@ check-run-results
//@ needs-subprocess

const C: &'static [u8; 5] = b"hello";

#[allow(unconditional_panic)]
fn test() -> u8 {
    C[10]
}

fn main() {
    test();
}
