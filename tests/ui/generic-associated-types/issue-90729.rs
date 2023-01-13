// check-pass

use std::marker::PhantomData;

pub trait Type {
    type Ref<'a>;
}

pub trait AsBytes {}

impl AsBytes for &str {}

pub struct Utf8;

impl Type for Utf8 {
    type Ref<'a> = &'a str;
}

pub struct Bytes<T: Type> {
    _marker: PhantomData<T>,
}

impl<T: Type> Bytes<T>
where
    for<'a> T::Ref<'a>: AsBytes,
{
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

fn main() {
    let _b = Bytes::<Utf8>::new();
}
