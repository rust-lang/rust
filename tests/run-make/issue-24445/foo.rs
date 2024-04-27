#![crate_type = "staticlib"]

struct Destroy;
impl Drop for Destroy {
    fn drop(&mut self) { println!("drop"); }
}

thread_local! {
    static X: Destroy = Destroy
}

#[no_mangle]
pub extern "C" fn foo() {
    X.with(|_| ());
}
