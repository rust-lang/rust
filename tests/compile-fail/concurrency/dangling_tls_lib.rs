// ignore-windows: Concurrency on Windows is not supported yet.

#![feature(thread_local_internals)]

use std::cell::RefCell;
use std::thread;

static A: std::thread::LocalKey<RefCell<u8>> = {
    #[inline]
    fn __init() -> RefCell<u8> {
        RefCell::new(0)
    }

    unsafe fn __getit() -> Option<&'static RefCell<u8>> {
        static __KEY: std::thread::__OsLocalKeyInner<RefCell<u8>> =
            std::thread::__OsLocalKeyInner::new();
        __KEY.get(__init)
    }

    unsafe { std::thread::LocalKey::new(__getit) }
};

struct Sender(*mut u8);

unsafe impl Send for Sender {}

fn main() {
    A.with(|f| {
        assert_eq!(*f.borrow(), 0);
        *f.borrow_mut() = 4;
    });

    let handle = thread::spawn(|| {
        let ptr = A.with(|f| {
            assert_eq!(*f.borrow(), 0);
            *f.borrow_mut() = 5;
            &mut *f.borrow_mut() as *mut u8
        });
        Sender(ptr)
    });
    let ptr = handle.join().unwrap().0;
    A.with(|f| {
        assert_eq!(*f.borrow(), 4);
    });
    let _x = unsafe { *ptr }; //~ ERROR Undefined Behavior
}
