pub trait Trait {
    fn create() -> impl Iterator<Item = u64> {
        std::iter::empty()
    }
}

pub struct Basic;
pub struct Intermediate;
pub struct Advanced;

impl Trait for Basic {
    // method provided by the trait
}

impl Trait for Intermediate {
    fn create() -> std::ops::Range<u64> { // concrete return type
        0..1
    }
}

impl Trait for Advanced {
    fn create() -> impl Iterator<Item = u64> { // opaque return type
        std::iter::repeat(0)
    }
}

// Regression test for issue #113929:

pub trait Def {
    fn def<T>() -> impl Default {}
}

impl Def for () {}
