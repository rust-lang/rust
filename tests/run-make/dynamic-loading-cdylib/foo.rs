#![crate_type = "cdylib"]

#[no_mangle]
pub extern "C" fn extern_fn_1(a: u32, b: u32) -> u32 {
    println!("extern_fn_1");
    a + b
}

#[derive(Default)]
struct NotifyOnDrop {
    last_result: std::cell::Cell<u32>,
}

impl NotifyOnDrop {
    fn save_and_print(&self, result: u32) {
        self.last_result.set(result);
    }
}

impl Drop for NotifyOnDrop {
    fn drop(&mut self) {
        println!("dropping, last result: {}", self.last_result.get());
    }
}

thread_local! {
    static FOO: NotifyOnDrop = NotifyOnDrop::default();
}

#[no_mangle]
pub extern "C" fn extern_fn_2(a: u32, b: u32) -> u32 {
    let result = a * b;

    FOO.with(|foo| {
        foo.save_and_print(result);
    });

    println!("extern_fn_2({a}, {b})");

    result
}
