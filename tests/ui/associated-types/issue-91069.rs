//@ check-pass

pub trait Associate {
    type Associated;
}

pub struct Wrap<'a> {
    pub field: &'a i32,
}

pub trait Create<T> {
    fn create() -> Self;
}

pub fn oh_no<'a, T>()
where
    Wrap<'a>: Associate,
    <Wrap<'a> as Associate>::Associated: Create<T>,
{
    <Wrap<'a> as Associate>::Associated::create();
}


pub fn main() {}
