#[no_mangle]
pub extern "C" fn hello() {
    println!("Hello world!");
}

#[no_mangle]
pub extern "C" fn number() -> u32 {
    42
}
