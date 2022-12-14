//@ignore-32bit

fn main() {
    let _fat: [u8; (1 << 61) + (1 << 31)];
    _fat = [0; (1u64 << 61) as usize + (1u64 << 31) as usize]; //~ ERROR: post-monomorphization error
}
