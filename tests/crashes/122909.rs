//@ compile-flags: -Zpolymorphize=on -Zinline-mir=yes
//@ known-bug: #122909


use std::sync::{Arc, Context, Weak};

pub struct WeakOnce<T>();
impl<T> WeakOnce<T> {
    extern "rust-call" fn try_get(&self) -> Option<Arc<T>> {}

    pub fn get(&self) -> Arc<T> {
        self.try_get()
            .unwrap_or_else(|| panic!("Singleton {} not available", std::any::type_name::<T>()))
    }
}
