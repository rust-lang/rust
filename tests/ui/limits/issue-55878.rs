//@ build-fail

fn main() {
    println!("Size: {}", std::mem::size_of::<[u8; u64::MAX as usize]>());
    //~^ ERROR evaluation of constant value failed
}
