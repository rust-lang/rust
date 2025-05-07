//@ check-pass
#![allow(type_alias_bounds)]


use std::marker::PhantomData;

pub struct Handle<T, I>(T, I);

impl<T, I> Handle<T, I> {
    pub fn get_info(&self) -> &I {
        let Handle(_, ref info) = *self;
        info
    }
}

pub struct BufferHandle<D: Device, T> {
    raw: RawBufferHandle<D>,
    _marker: PhantomData<T>,
}

impl<D: Device, T> BufferHandle<D, T> {
    pub fn get_info(&self) -> &String {
        self.raw.get_info()
    }
}

pub type RawBufferHandle<D: Device> = Handle<<D as Device>::Buffer, String>;

pub trait Device {
    type Buffer;
}

fn main() {}
