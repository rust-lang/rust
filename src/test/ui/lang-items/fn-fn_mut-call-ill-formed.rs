// Make sure that an error is reported if the `call` function of the
// `fn`/`fn_mut` lang item is grossly ill-formed.

#![feature(lang_items)]
#![feature(no_core)]
#![no_core]

#[lang = "fn"]
trait MyFn<T> {
    const call: i32 = 42;
    //~^ ERROR: `call` trait item in `fn` lang item must be a function
}

#[lang = "fn_mut"]
trait MyFnMut<T> {
    fn call(i: i32, j: i32) -> i32 { i + j }
    //~^ ERROR: first argument of `call` in `fn_mut` lang item must be a reference
}

fn main() {
    let a = || 42;
    a();

    let mut i = 0;
    let mut b = || { i += 1; };
    b();
}
