// ignore-windows: Unwind panicking does not currently work on Windows
#![feature(never_type)]
#![allow(const_err)]
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::cell::Cell;

thread_local! {
    static MY_COUNTER: Cell<usize> = Cell::new(0);
    static DROPPED: Cell<bool> = Cell::new(false);
    static HOOK_CALLED: Cell<bool> = Cell::new(false);
}

struct DropTester;

impl Drop for DropTester {
    fn drop(&mut self) {
        DROPPED.with(|c| {
            c.set(true);
        });
    }
}

fn do_panic_counter(do_panic: impl FnOnce(usize) -> !) {
    // If this gets leaked, it will be easy to spot
    // in Miri's leak report
    let _string = "LEAKED FROM do_panic_counter".to_string();

    // When we panic, this should get dropped during unwinding
    let _drop_tester = DropTester;

    // Check for bugs in Miri's panic implementation.
    // If do_panic_counter() somehow gets called more than once,
    // we'll generate a different panic message and stderr will differ.
    let old_val = MY_COUNTER.with(|c| {
        let val = c.get();
        c.set(val + 1);
        val
    });
    do_panic(old_val);
}

fn main() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        HOOK_CALLED.with(|h| h.set(true));
        prev(panic_info)
    }));

    // Std panics
    test(|_old_val| std::panic!("Hello from panic: std"));
    test(|old_val| std::panic!(format!("Hello from panic: {:?}", old_val)));
    test(|old_val| std::panic!("Hello from panic: {:?}", old_val));
    test(|_old_val| std::panic!(1337));

    // Core panics
    test(|_old_val| core::panic!("Hello from panic: core"));
    test(|old_val| core::panic!(&format!("Hello from panic: {:?}", old_val)));
    test(|old_val| core::panic!("Hello from panic: {:?}", old_val));

    // Built-in panics
    test(|_old_val| { let _val = [0, 1, 2][4]; loop {} });
    test(|_old_val| { let _val = 1/0; loop {} });

    // Cleanup: reset to default hook.
    drop(std::panic::take_hook());

    eprintln!("Success!"); // Make sure we get this in stderr
}

fn test(do_panic: impl FnOnce(usize) -> !) {
    // Reset test flags.
    DROPPED.with(|c| c.set(false));
    HOOK_CALLED.with(|c| c.set(false));

    // Cause and catch a panic.
    let res = catch_unwind(AssertUnwindSafe(|| {
        let _string = "LEAKED FROM CLOSURE".to_string();
        do_panic_counter(do_panic)
    })).expect_err("do_panic() did not panic!");

    // See if we can extract the panic message.
    if let Some(s) = res.downcast_ref::<String>() {
        eprintln!("Caught panic message (String): {}", s);
    } else if let Some(s) = res.downcast_ref::<&str>() {
        eprintln!("Caught panic message (&str): {}", s);
    } else {
        eprintln!("Failed get caught panic message.");
    }

    // Test flags.
    assert!(DROPPED.with(|c| c.get()));
    assert!(HOOK_CALLED.with(|c| c.get()));
}

