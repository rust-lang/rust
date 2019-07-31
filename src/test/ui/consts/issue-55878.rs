// error-pattern: reaching this expression at runtime will panic or abort
fn main() {
    println!("Size: {}", std::mem::size_of::<[u8; std::u64::MAX as usize]>());
}
