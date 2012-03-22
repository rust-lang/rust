


// Regression test for issue #152.
fn main() {
    let mut b: uint = 1u;
    while b <= 32u {
        0u << b;
        b <<= 1u;
        log(debug, b);
    }
}
