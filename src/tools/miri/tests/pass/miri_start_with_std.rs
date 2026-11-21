#![no_main]

#[no_mangle]
fn miri_start(argc: isize, argv: *const *const u8) -> isize {
    // Ensure argument array is null-terminated.
    assert!(unsafe { (*argv.add(argc as usize)).is_null() });

    // Ensure basic std functionality works.
    let _b = Box::new(0);
    println!("hello, world!");
    0
}
