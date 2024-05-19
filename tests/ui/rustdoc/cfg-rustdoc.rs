#[cfg(doc)]
pub struct Foo;

fn main() {
    let f = Foo; //~ ERROR
}
