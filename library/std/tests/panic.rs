#![allow(dead_code)]

use core::panic::PanicInfo;
use std::cell::RefCell;
use std::panic::{AssertUnwindSafe, Location, PanicHookInfo, UnwindSafe};
use std::rc::Rc;
use std::sync::{Arc, Mutex, RwLock};

struct Foo {
    a: i32,
}

fn assert<T: UnwindSafe + ?Sized>() {}

#[test]
fn panic_safety_traits() {
    assert::<i32>();
    assert::<&i32>();
    assert::<*mut i32>();
    assert::<*const i32>();
    assert::<usize>();
    assert::<str>();
    assert::<&str>();
    assert::<Foo>();
    assert::<&Foo>();
    assert::<Vec<i32>>();
    assert::<String>();
    assert::<RefCell<i32>>();
    assert::<Box<i32>>();
    assert::<Mutex<i32>>();
    assert::<RwLock<i32>>();
    assert::<&Mutex<i32>>();
    assert::<&RwLock<i32>>();
    assert::<Rc<i32>>();
    assert::<Arc<i32>>();
    assert::<Box<[u8]>>();

    {
        trait Trait: UnwindSafe {}
        assert::<Box<dyn Trait>>();
    }

    fn bar<T>() {
        assert::<Mutex<T>>();
        assert::<RwLock<T>>();
    }

    fn baz<T: UnwindSafe>() {
        assert::<Box<T>>();
        assert::<Vec<T>>();
        assert::<RefCell<T>>();
        assert::<AssertUnwindSafe<T>>();
        assert::<&AssertUnwindSafe<T>>();
        assert::<Rc<AssertUnwindSafe<T>>>();
        assert::<Arc<AssertUnwindSafe<T>>>();
    }
}

#[test]
fn panic_info_static_location<'x>() {
    // Verify that the returned `Location<'_>`s generic lifetime is 'static when
    // calling `PanicInfo::location`. Test failure is indicated by a compile
    // failure, not a runtime panic.
    let _: for<'a> fn(&'a PanicInfo<'x>) -> Option<&'a Location<'static>> = PanicInfo::location;
}

#[test]
fn panic_hook_info_static_location<'x>() {
    // Verify that the returned `Location<'_>`s generic lifetime is 'static when
    // calling `PanicHookInfo::location`. Test failure is indicated by a compile
    // failure, not a runtime panic.
    let _: for<'a> fn(&'a PanicHookInfo<'x>) -> Option<&'a Location<'static>> =
        PanicHookInfo::location;
}
