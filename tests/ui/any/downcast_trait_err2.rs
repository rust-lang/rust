//@ run-pass
#![feature(downcast_trait)]
use std::{any::downcast_trait, sync::OnceLock};

trait Trait<T> {
    fn call(&self, t: T, x: &Box<i32>);
}

impl Trait<for<'a> fn(&'a Box<i32>)> for () {
    fn call(&self, f: for<'a> fn(&'a Box<i32>), x: &Box<i32>) {
        f(x);
    }
}

static STORAGE: OnceLock<&'static Box<i32>> = OnceLock::new();

fn store(x: &'static Box<i32>) {
    STORAGE.set(x).unwrap();
}

fn main() {
    let data = Box::new(Box::new(1i32));
    downcast_trait::<_, dyn Trait<fn(&'static Box<i32>)>>(&())
        .unwrap()
        .call(store, &*data);
    drop(data);
    println!("{}", STORAGE.get().unwrap());
}
