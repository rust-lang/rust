#[cfg(rustdoc)]
pub struct Foo;

fn main() {
    let f = Foo; //~ ERROR
}
