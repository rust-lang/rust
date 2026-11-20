//@ compile-flags: -Cmetadata=aux

pub trait Trait {
    type Output;
}

pub fn fun<T>(_: T) where T: Trait<Output=i32> {}
