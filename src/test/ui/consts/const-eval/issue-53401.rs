// build-pass (FIXME(62277): could be check-pass?)

pub const STATIC_TRAIT: &dyn Test = &();

fn main() {}

pub trait Test {
    fn test() where Self: Sized {}
}

impl Test for () {}
