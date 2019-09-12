fn main() {
    const N: usize = 10;

    let x = vec![0x4141u16; N];

    let mut y: Vec<u8> = unsafe { std::mem::transmute(x) };
    unsafe { y.set_len(2 * N) };

    println!("{:?}", std::str::from_utf8(&y).unwrap());

    let mut x: Vec<u16> = unsafe { std::mem::transmute(y) };
    unsafe { x.set_len(N) };
}
