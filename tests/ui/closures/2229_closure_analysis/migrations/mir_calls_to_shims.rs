//@ run-rustfix
//@ needs-unwind

#![deny(rust_2021_incompatible_closure_captures)]
//~^ NOTE: the lint level is defined here
#![feature(fn_traits)]
#![feature(never_type)]

use std::panic;

fn foo_diverges() -> ! {
    panic!()
}

fn assert_panics<F>(f: F)
where
    F: FnOnce(),
{
    let f = panic::AssertUnwindSafe(f);
    let result = panic::catch_unwind(move || {
        //~^ ERROR: changes to closure capture in Rust 2021 will affect which traits the closure implements [rust_2021_incompatible_closure_captures]
        //~| NOTE: in Rust 2018, this closure implements `UnwindSafe`
        //~| NOTE: in Rust 2018, this closure implements `RefUnwindSafe`
        //~| NOTE: for more information, see
        //~| HELP: add a dummy let to cause `f` to be fully captured
        f.0()
        //~^ NOTE: in Rust 2018, this closure captures all of `f`, but in Rust 2021, it will only capture `f.0`
    });
    if let Ok(..) = result {
        panic!("diverging function returned");
    }
}

fn test_fn_ptr_panic<T>(mut t: T)
where
    T: Fn() -> !,
{
    let as_fn = <T as Fn<()>>::call;
    assert_panics(|| as_fn(&t, ()));
    let as_fn_mut = <T as FnMut<()>>::call_mut;
    assert_panics(|| as_fn_mut(&mut t, ()));
    let as_fn_once = <T as FnOnce<()>>::call_once;
    assert_panics(|| as_fn_once(t, ()));
}

fn main() {
    test_fn_ptr_panic(foo_diverges);
    test_fn_ptr_panic(foo_diverges as fn() -> !);
}
