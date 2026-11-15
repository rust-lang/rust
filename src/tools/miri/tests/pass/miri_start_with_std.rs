#![no_main]

#[no_mangle]
fn miri_start(argc: isize, argv: *const *const u8) -> isize {
    let _b = Box::new(0);
    println!("hello, world!");
    assert!(unsafe { (*argv.add(argc as usize)).is_null() });
    0
}
