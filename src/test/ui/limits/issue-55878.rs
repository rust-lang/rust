// build-fail
// normalize-stderr-64bit "18446744073709551615" -> "SIZE"
// normalize-stderr-32bit "4294967295" -> "SIZE"

// error-pattern: are too big for the current architecture
fn main() {
    println!("Size: {}", std::mem::size_of::<[u8; u64::MAX as usize]>());
}
