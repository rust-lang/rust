#[link(name = "test")]
extern "C" {
    fn foo() -> i32;
}

fn main() {
    unsafe {
        foo();
    }
}
