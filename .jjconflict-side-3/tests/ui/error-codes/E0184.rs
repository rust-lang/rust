#[derive(Copy)]
struct Foo; //~ ERROR E0184

impl Drop for Foo {
    fn drop(&mut self) {
    }
}

fn main() {
}
