// mac-os `getrandom_1` does some pointer shenanigans
//@compile-flags: -Zmiri-permissive-provenance

/// Test old version of `getrandom`.
fn main() {
    let mut data = vec![0; 16];
    getrandom_1::getrandom(&mut data).unwrap();
}
