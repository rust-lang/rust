pub trait Meow {
    fn meow(&self) {}
}

pub struct GlobalMeow;

impl Meow for GlobalMeow {}

pub(crate) struct PrivateMeow;

impl Meow for PrivateMeow {}
