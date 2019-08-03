// error-pattern: the type `[u8; 18446744073709551615]` is too big for the current architecture
fn main() {
    println!("Size: {}", std::mem::size_of::<[u8; std::u64::MAX as usize]>());
}
