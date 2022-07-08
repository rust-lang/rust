#![allow(dead_code)]

use crate::cell::RefCell;
use crate::panic::{AssertUnwindSafe, UnwindSafe};
use crate::rc::Rc;
use crate::sync::{Arc, Mutex, RwLock};

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
fn test_try_panic_any_message_drop_glue_does_happen() {
    use crate::sync::Arc;

    let count = Arc::new(());
    let weak = Arc::downgrade(&count);

    match super::catch_unwind(|| super::panic_any(count)) {
        Ok(()) => panic!("closure did not panic"),
        Err(e) if e.is::<Arc<()>>() => {}
        Err(_) => panic!("closure did not panic with the expected payload"),
    }
    assert!(weak.upgrade().is_none());
}

#[test]
fn test_try_panic_resume_unwind_drop_glue_does_happen() {
    use crate::sync::Arc;

    let count = Arc::new(());
    let weak = Arc::downgrade(&count);

    match super::catch_unwind(|| super::resume_unwind(Box::new(count))) {
        Ok(()) => panic!("closure did not panic"),
        Err(e) if e.is::<Arc<()>>() => {}
        Err(_) => panic!("closure did not panic with the expected payload"),
    }
    assert!(weak.upgrade().is_none());
}
