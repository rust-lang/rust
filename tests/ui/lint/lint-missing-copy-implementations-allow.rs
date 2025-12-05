//@ check-pass
#![deny(missing_copy_implementations)]

// Don't recommend implementing Copy on something stateful like an iterator.
pub struct MyIterator {
    num: u8,
}

impl Iterator for MyIterator {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

pub struct Handle {
    inner: *mut (),
}

pub struct Handle2 {
    inner: *const (),
}

pub enum MaybeHandle {
    Ptr(*mut ()),
}

pub union UnionHandle {
    ptr: *mut (),
}

pub struct Array([u8; 2048]);

fn main() {}
