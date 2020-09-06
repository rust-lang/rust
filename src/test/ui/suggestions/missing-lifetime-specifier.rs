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
    //~^ ERROR wrong number of lifetime arguments: expected 2, found 1
    //~| ERROR wrong number of lifetime arguments: expected 2, found 1
    //~| ERROR wrong number of lifetime arguments: expected 2, found 1
    //~| ERROR wrong number of lifetime arguments: expected 2, found 1
}
thread_local! {
    static f: RefCell<HashMap<i32, Vec<Vec<&Tar<'static, i32>>>>> = RefCell::new(HashMap::new());
    //~^ ERROR wrong number of lifetime arguments: expected 2, found 1
    //~| ERROR wrong number of lifetime arguments: expected 2, found 1
    //~| ERROR wrong number of lifetime arguments: expected 2, found 1
    //~| ERROR wrong number of lifetime arguments: expected 2, found 1
    //~| ERROR missing lifetime specifier
    //~| ERROR missing lifetime specifier
}

fn main() {}
