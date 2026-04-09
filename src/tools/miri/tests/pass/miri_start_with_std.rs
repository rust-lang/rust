#![no_main]

#[no_mangle]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    let _b = Box::new(0);
    println!("hello, world!");
    0
}
