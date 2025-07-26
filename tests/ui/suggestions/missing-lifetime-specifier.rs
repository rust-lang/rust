// The specific errors produced depend the thread-local implementation.
// Run only on platforms with "fast" TLS.
//@ ignore-wasm globals are used instead of thread locals
//@ ignore-emscripten globals are used instead of thread locals
//@ ignore-android does not use #[thread_local]
//@ ignore-nto does not use #[thread_local]
// Different number of duplicated diagnostics on different targets
//@ compile-flags: -Zdeduplicate-diagnostics=yes

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
    static a: RefCell<HashMap<i32, Vec<Vec<Foo>>>> = RefCell::new(HashMap::new());
    //~^ ERROR missing lifetime specifiers
}
thread_local! {
    static b: RefCell<HashMap<i32, Vec<Vec<&dyn Bar>>>> = RefCell::new(HashMap::new());
    //~^ ERROR missing lifetime specifiers
}
thread_local! {
    static c: RefCell<HashMap<i32, Vec<Vec<Qux<i32>>>>> = RefCell::new(HashMap::new());
    //~^ ERROR missing lifetime specifiers
}
thread_local! {
    static d: RefCell<HashMap<i32, Vec<Vec<&dyn Tar<i32>>>>> = RefCell::new(HashMap::new());
    //~^ ERROR missing lifetime specifiers
}

thread_local! {
    static e: RefCell<HashMap<i32, Vec<Vec<Qux<'static, i32>>>>> = RefCell::new(HashMap::new());
    //~^ ERROR union takes 2 lifetime arguments but 1 lifetime argument
}
thread_local! {
    static f: RefCell<HashMap<i32, Vec<Vec<&dyn Tar<'static, i32>>>>> =
        RefCell::new(HashMap::new());
    //~^^ ERROR trait takes 2 lifetime arguments but 1 lifetime argument was supplied
    //~| ERROR missing lifetime specifier
}

fn main() {}
