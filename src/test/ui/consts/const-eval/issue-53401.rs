// compile-pass

pub const STATIC_TRAIT: &Test = &();

fn main() {}

pub trait Test {
    fn test() where Self: Sized {}
}

impl Test for () {}
