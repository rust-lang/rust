#![crate_type = "rlib"]

#[link(name = "foo", kind = "static")]
extern "C" {
    fn foo();
    fn bar();
}

pub fn baz() {
    unsafe {
        foo();
        bar();
    }
}
