#![feature(const_trait_impl)]
#![feature(const_destruct)]

use std::marker::Destruct;

struct NotConstDrop;

impl Drop for NotConstDrop {
    fn drop(&mut self) {}
}

struct ConstDrop(NotConstDrop);
//~^ ERROR: `NotConstDrop` does not implement `[const] Destruct`

const impl Drop for ConstDrop {
    fn drop(&mut self) {}
}

struct ConstDrop2<T>(T);
//~^ ERROR:  `T` does not implement `[const] Destruct`

const impl<T> Drop for ConstDrop2<T> {
    fn drop(&mut self) {}
}

struct ConstDrop3<T>(T);

const impl<T: [const] Destruct> Drop for ConstDrop3<T> {
    fn drop(&mut self) {}
}

fn main() {}
