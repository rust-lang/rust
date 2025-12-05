//@ run-fail
//@ error-pattern:index out of bounds: the len is 5 but the index is 10
//@ needs-subprocess

const C: &'static [u8; 5] = b"hello";

#[allow(unconditional_panic)]
fn test() -> u8 {
    C[10]
}

fn main() {
    test();
}
