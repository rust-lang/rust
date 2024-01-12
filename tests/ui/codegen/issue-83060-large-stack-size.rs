// build-fail
fn func() {
    const CAP: usize = std::u32::MAX as usize;
    let mut x: [u8; CAP>>1] = [0; CAP>>1];
    x[2] = 123;
    println!("{}", x[2]);
}

fn main() {
    let mut n = 5;
    if cfg!(target_pointer_width = "32") {
        n = 3;
    }
    std::thread::Builder::new()
        .stack_size(n * 1024 * 1024 * 1024)
        .spawn(func)
        .unwrap()
        .join()
        .unwrap();
}
