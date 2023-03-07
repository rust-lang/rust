pub trait Trait {
    fn dummy(&self) { }
}

pub struct Foo<T:Trait> {
    pub x: T,
}

pub enum Bar<T:Trait> {
    ABar(isize),
    BBar(T),
    CBar(usize),
}
