// mac-os `getrandom_01` does some pointer shenanigans
//@compile-flags: -Zmiri-permissive-provenance

/// Test direct calls of getrandom 0.1 and 0.2
fn main() {
    let mut data = vec![0; 16];
    getrandom_01::getrandom(&mut data).unwrap();
    getrandom_02::getrandom(&mut data).unwrap();
}
