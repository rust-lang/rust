#[macro_use]
extern crate foo2; // foo2 first to exhibit the bug
#[macro_use]
extern crate foo1;

fn main() {
    foo2::foo2(foo1::A);
}
