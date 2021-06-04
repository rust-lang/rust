// run-rustfix

#![deny(disjoint_capture_migration)]
// ignore-wasm32-bare compiled with panic=abort by default

#![feature(fn_traits)]
#![feature(never_type)]

use std::panic;

fn foo_diverges() -> ! { panic!() }

fn assert_panics<F>(f: F) where F: FnOnce() {
    let f = panic::AssertUnwindSafe(f);
    let result = panic::catch_unwind(move || {
        //~^ ERROR: `UnwindSafe`, `RefUnwindSafe` trait implementation affected for closure because of `capture_disjoint_fields`
        //~| HELP: add a dummy let to cause `f` to be fully captured
        f.0()
    });
    if let Ok(..) = result {
        panic!("diverging function returned");
    }
}

fn test_fn_ptr_panic<T>(mut t: T)
    where T: Fn() -> !
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
