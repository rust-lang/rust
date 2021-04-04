#![allow(bare_trait_objects)]
use std::collections::HashMap;
use std::cell::RefCell;

pub union Foo<'t, 'k> {
    i: &'t i64,
    f: &'k f64,
}
trait Bar<'t, 'k> {}

pub union Qux<'t, 'k, I> {
    i: &'t I,
    f: &'k I,
}
trait Tar<'t, 'k, I> {}

thread_local! {
    static a: RefCell<HashMap<i32, Vec<Vec<Foo>>>> = RefCell::new(HashMap::new());
    //~^ ERROR missing lifetime specifier
    //~| ERROR missing lifetime specifier
}
thread_local! {
    static b: RefCell<HashMap<i32, Vec<Vec<&Bar>>>> = RefCell::new(HashMap::new());
    //~^ ERROR missing lifetime specifier
    //~| ERROR missing lifetime specifier
    //~| ERROR missing lifetime specifier
    //~| ERROR missing lifetime specifier
}
thread_local! {
    static c: RefCell<HashMap<i32, Vec<Vec<Qux<i32>>>>> = RefCell::new(HashMap::new());
    //~^ ERROR missing lifetime specifier
    //~| ERROR missing lifetime specifier
}
thread_local! {
    static d: RefCell<HashMap<i32, Vec<Vec<&Tar<i32>>>>> = RefCell::new(HashMap::new());
    //~^ ERROR missing lifetime specifier
    //~| ERROR missing lifetime specifier
    //~| ERROR missing lifetime specifier
    //~| ERROR missing lifetime specifier
}

thread_local! {
    static e: RefCell<HashMap<i32, Vec<Vec<Qux<'static, i32>>>>> = RefCell::new(HashMap::new());
    //~^ ERROR this union takes 2 lifetime arguments but only 1 lifetime argument was supplied
    //~| ERROR this union takes 2 lifetime arguments but only 1 lifetime argument was supplied
    //~| ERROR this union takes 2 lifetime arguments but only 1 lifetime argument was supplied
    //~| ERROR this union takes 2 lifetime arguments but only 1 lifetime argument was supplied
}
thread_local! {
    static f: RefCell<HashMap<i32, Vec<Vec<&Tar<'static, i32>>>>> = RefCell::new(HashMap::new());
    //~^ ERROR this trait takes 2 lifetime arguments but only 1 lifetime argument was supplied
    //~| ERROR this trait takes 2 lifetime arguments but only 1 lifetime argument was supplied
    //~| ERROR this trait takes 2 lifetime arguments but only 1 lifetime argument was supplied
    //~| ERROR this trait takes 2 lifetime arguments but only 1 lifetime argument was supplied
    //~| ERROR missing lifetime specifier
    //~| ERROR missing lifetime specifier
}

fn main() {}
