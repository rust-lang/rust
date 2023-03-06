// revisions: fn_once_bad_item fn_once_bad_sig fn_mut_bad_item fn_mut_bad_sig fn_bad_item fn_bad_sig

#![feature(lang_items)]
#![feature(no_core)]
#![no_core]

#[lang = "sized"]
trait Sized {}

#[cfg(any(fn_bad_item, fn_bad_sig))]
#[lang = "fn"]
trait MyFn<T> {
    #[cfg(fn_bad_sig)]
    fn call(i: i32) -> i32 { 0 }

    #[cfg(fn_bad_item)]
    const call: i32 = 42;
}

#[cfg(any(fn_mut_bad_item, fn_mut_bad_sig))]
#[lang = "fn_mut"]
trait MyFnMut<T> {
    #[cfg(fn_mut_bad_sig)]
    fn call_mut(i: i32) -> i32 { 0 }

    #[cfg(fn_mut_bad_item)]
    const call_mut: i32 = 42;
}

#[cfg(any(fn_once_bad_item, fn_once_bad_sig))]
#[lang = "fn_once"]
trait MyFnOnce<T> {
    #[cfg(fn_once_bad_sig)]
    fn call_once(i: i32) -> i32 { 0 }

    #[cfg(fn_once_bad_item)]
    const call_once: i32 = 42;
}

fn main() {
    let a = || 42;
    a();
    //~^ ERROR failed to find an overloaded call trait for closure call

    let mut i = 0;
    let mut b = || { };
    b();
    //~^ ERROR failed to find an overloaded call trait for closure call
}
