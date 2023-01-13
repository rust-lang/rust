// run-pass

use std::cell::UnsafeCell;
use std::collections::HashMap;

struct OnceCell<T> {
    _value: UnsafeCell<Option<T>>,
}

impl<T> OnceCell<T> {
    const INIT: OnceCell<T> = OnceCell {
        _value: UnsafeCell::new(None),
    };
}

pub fn crash<K, T>() {
    let _ = OnceCell::<HashMap<K, T>>::INIT;
}

fn main() {}
