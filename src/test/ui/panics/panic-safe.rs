// run-pass
#![allow(dead_code)]

use std::panic::{UnwindSafe, AssertUnwindSafe};
use std::cell::RefCell;
use std::sync::{Mutex, RwLock, Arc};
use std::rc::Rc;

struct Foo { a: i32 }

fn assert<T: UnwindSafe + ?Sized>() {}

fn main() {
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

    trait Trait: UnwindSafe {}
    assert::<Box<dyn Trait>>();

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
