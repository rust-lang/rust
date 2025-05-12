use std::hash::BuildHasher;

fn next_u64() -> u64 {
    let bh = std::hash::RandomState::new();
    let h = bh.build_hasher();
    h.finish() //~ ERROR no method named `finish` found for struct `DefaultHasher`
}

trait Bar {}
impl Bar for String {}

fn main() {
    let s = String::from("hey");
    let x: &dyn Bar = &s;
    x.as_ref(); //~ ERROR the method `as_ref` exists for reference `&dyn Bar`, but its trait bounds
}
