// check-pass

use std::marker::PhantomData;

pub struct MyGenericType<T> {
    _marker: PhantomData<*const T>,
}

pub struct MyNonGenericType;

impl<T> From<MyGenericType<T>> for MyNonGenericType {
    fn from(_: MyGenericType<T>) -> Self {
        todo!()
    }
}

pub trait MyTrait {
    const MY_CONSTANT: i32;
}

impl<T> MyTrait for MyGenericType<T>
where
    Self: Into<MyNonGenericType>,
{
    const MY_CONSTANT: i32 = 1;
}

impl<T> MyGenericType<T> {
    const MY_OTHER_CONSTANT: i32 = <MyGenericType<T> as MyTrait>::MY_CONSTANT;
}

fn main() {}
