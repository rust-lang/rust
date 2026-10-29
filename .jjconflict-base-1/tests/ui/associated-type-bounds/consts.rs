pub fn accept(_: impl Trait<K: Copy>) {}
//~^ ERROR expected type, found constant

pub trait Trait {
    const K: i32;
}

fn main() {}
