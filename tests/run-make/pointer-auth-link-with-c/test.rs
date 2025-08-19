#[link(name = "test", kind = "static")]
extern "C" {
    fn foo() -> i32;
}

fn main() {
    unsafe {
        foo();
    }
}
