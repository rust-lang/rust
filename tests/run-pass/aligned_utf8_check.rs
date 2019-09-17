fn main() {
    const N: usize = 10;
    let vec = vec![0x4141414141414141u64; N];
    let content = unsafe { std::slice::from_raw_parts(vec.as_ptr() as *const u8, 8 * N) };
    println!("{:?}", std::str::from_utf8(content).unwrap());
}
