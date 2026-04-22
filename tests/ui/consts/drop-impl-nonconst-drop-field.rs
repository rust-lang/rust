#![feature(const_trait_impl)]
#![feature(const_destruct)]

use std::marker::Destruct;

struct NotConstDrop;

impl Drop for NotConstDrop {
    fn drop(&mut self) {}
}

struct ConstDrop(NotConstDrop);
//~^ ERROR: `NotConstDrop` does not implement `[const] Destruct`

impl const Drop for ConstDrop {
    fn drop(&mut self) {}
}

struct ConstDrop2<T>(T);
//~^ ERROR:  `T` does not implement `[const] Destruct`

impl<T> const Drop for ConstDrop2<T> {
    fn drop(&mut self) {}
}

struct ConstDrop3<T>(T);

impl<T: [const] Destruct> const Drop for ConstDrop3<T> {
    fn drop(&mut self) {}
}

fn main() {}
