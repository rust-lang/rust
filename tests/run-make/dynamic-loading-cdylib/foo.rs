#![crate_type = "cdylib"]

#[no_mangle]
pub extern "C" fn extern_fn_1(a: u32, b: u32) -> u32 {
    println!("extern_fn_1");
    a + b
}

struct NotifyOnDrop;

impl Drop for NotifyOnDrop {
    fn drop(&mut self) {
        println!("drop");
    }
}

#[no_mangle]
pub extern "C" fn extern_fn_2(a: u32, b: u32) -> u32 {
    thread_local!(static FOO: NotifyOnDrop = NotifyOnDrop);
    FOO.with(|_foo| {});
    println!("extern_fn_2");
    a * b
}
