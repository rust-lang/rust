// This test checks that allocating a stack size of 1GB or more results in a warning
//@build-pass
//@ only-64bit

fn func() {
    const CAP: usize = std::u32::MAX as usize;
    let mut x: [u8; CAP>>1] = [0; CAP>>1];
    //~^ warning: allocation of size: 1 GiB  exceeds most system architecture limits
    //~| NOTE on by default
    x[2] = 123;
    println!("{}", x[2]);
}

fn main() {
    std::thread::Builder::new()
        .stack_size(3 * 1024 * 1024 * 1024)
        .spawn(func)
        .unwrap()
        .join()
        .unwrap();
}
