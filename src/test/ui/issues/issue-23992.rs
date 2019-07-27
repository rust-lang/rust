// run-pass
pub struct Outer<T: Trait>(T);
pub struct Inner<'a> { value: &'a bool }

pub trait Trait {
    type Error;
    fn ready(self) -> Self::Error;
}

impl<'a> Trait for Inner<'a> {
    type Error = Outer<Inner<'a>>;
    fn ready(self) -> Outer<Inner<'a>> { Outer(self) }
}

fn main() {
    let value = true;
    let inner = Inner { value: &value };
    assert_eq!(inner.ready().0.value, &value);
}
