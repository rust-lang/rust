// different number of duplicated diagnostics on different targets
// only-x86_64
// only-linux
// compile-flags: -Zdeduplicate-diagnostics=yes

#![allow(bare_trait_objects)]
use std::cell::RefCell;
use std::collections::HashMap;

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
    //~^ ERROR lifetime may not live long enough
    //~| ERROR lifetime may not live long enough
    static a: RefCell<HashMap<i32, Vec<Vec<Foo>>>> = RefCell::new(HashMap::new());
      //~^ ERROR missing lifetime specifiers
      //~| ERROR missing lifetime specifiers
}
thread_local! {
    //~^ ERROR lifetime may not live long enough
    //~| ERROR lifetime may not live long enough
    //~| ERROR lifetime may not live long enough
    static b: RefCell<HashMap<i32, Vec<Vec<&Bar>>>> = RefCell::new(HashMap::new());
      //~^ ERROR missing lifetime specifiers
      //~| ERROR missing lifetime specifiers
}
thread_local! {
    //~^ ERROR lifetime may not live long enough
    //~| ERROR lifetime may not live long enough
    static c: RefCell<HashMap<i32, Vec<Vec<Qux<i32>>>>> = RefCell::new(HashMap::new());
    //~^ ERROR missing lifetime specifiers
    //~| ERROR missing lifetime specifiers
}
thread_local! {
    //~^ ERROR lifetime may not live long enough
    //~| ERROR lifetime may not live long enough
    //~| ERROR lifetime may not live long enough
    static d: RefCell<HashMap<i32, Vec<Vec<&Tar<i32>>>>> = RefCell::new(HashMap::new());
    //~^ ERROR missing lifetime specifiers
    //~| ERROR missing lifetime specifiers
}

thread_local! {
    static e: RefCell<HashMap<i32, Vec<Vec<Qux<'static, i32>>>>> = RefCell::new(HashMap::new());
    //~^ ERROR union takes 2 lifetime arguments but 1 lifetime argument
}
thread_local! {
    static f: RefCell<HashMap<i32, Vec<Vec<&Tar<'static, i32>>>>> = RefCell::new(HashMap::new());
    //~^ ERROR trait takes 2 lifetime arguments but 1 lifetime argument was supplied
    //~| ERROR missing lifetime
    //~| ERROR missing lifetime
}

fn main() {}
