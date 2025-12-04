//@ build-fail

fn main() {
    println!("Size: {}", std::mem::size_of::<[u8; u64::MAX as usize]>());
    //~^ ERROR too big for the target architecture
}
