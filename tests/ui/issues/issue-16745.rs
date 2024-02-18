//@ run-pass
fn main() {
    const X: u8 = 0;
    let out: u8 = match 0u8 {
        X => 99,
        b'\t' => 1,
        1u8 => 2,
        _ => 3,
    };
    assert_eq!(out, 99);
}
