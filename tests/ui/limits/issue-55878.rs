//@ build-fail

//@ error-pattern: are too big for the target architecture
fn main() {
    println!("Size: {}", std::mem::size_of::<[u8; u64::MAX as usize]>());
}
