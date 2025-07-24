//@ run-pass
#![feature(downcast_trait)]

use std::{any::downcast_trait, sync::OnceLock};

trait Trait {
    fn call(&self, x: &Box<i32>);
}

impl Trait for for<'a> fn(&'a Box<i32>) {
    fn call(&self, x: &Box<i32>) {
        self(x);
    }
}

static STORAGE: OnceLock<&'static Box<i32>> = OnceLock::new();

fn store(x: &'static Box<i32>) {
    STORAGE.set(x).unwrap();
}

fn main() {
    let data = Box::new(Box::new(1i32));
    let fn_ptr: fn(&'static Box<i32>) = store;
    downcast_trait::<_, dyn Trait>(&fn_ptr)
        .unwrap()
        .call(&*data);
    drop(data);
    println!("{}", STORAGE.get().unwrap());
}
