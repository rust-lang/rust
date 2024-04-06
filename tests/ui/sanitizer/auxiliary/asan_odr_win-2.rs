//@ no-prefer-dynamic
//@ compile-flags: -Cunsafe-allow-abi-mismatch=sanitize -Zunstable-options -Zsanitizer=address
#![crate_name = "othercrate"]
#![crate_type = "rlib"]

pub fn exposed_func() {
    let result = std::panic::catch_unwind(|| {
        println!("hello!");
    });
    assert!(result.is_ok());
}
