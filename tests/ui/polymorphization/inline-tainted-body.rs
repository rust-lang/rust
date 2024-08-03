//@ compile-flags: -Zvalidate-mir -Zinline-mir=yes

#![feature(unboxed_closures)]

use std::sync::Arc;

pub struct WeakOnce<T>();
//~^ ERROR type parameter `T` is never used

impl<T> WeakOnce<T> {
    extern "rust-call" fn try_get(&self) -> Option<Arc<T>> {}
    //~^ ERROR functions with the "rust-call" ABI must take a single non-self tuple argument
    //~| ERROR mismatched types

    pub fn get(&self) -> Arc<T> {
        self.try_get()
            .unwrap_or_else(|| panic!("Singleton {} not available", std::any::type_name::<T>()))
    }
}

fn main() {}
