#[derive(Copy)] //~ ERROR E0184
struct Foo;

impl Drop for Foo {
    fn drop(&mut self) {
    }
}

fn main() {
}
