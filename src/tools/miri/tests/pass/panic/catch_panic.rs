// We test the `align_offset` panic below, make sure we test the interpreter impl and not the "real" one.
//@compile-flags: -Zmiri-symbolic-alignment-check -Zmiri-permissive-provenance
#![feature(never_type)]
#![allow(unconditional_panic, non_fmt_panics)]

use std::cell::Cell;
use std::panic::{catch_unwind, AssertUnwindSafe};

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
    test(None, |_old_val| std::panic!("Hello from panic: std"));
    test(None, |old_val| std::panic!(format!("Hello from panic: {:?}", old_val)));
    test(None, |old_val| std::panic!("Hello from panic: {:?}", old_val));
    test(None, |_old_val| std::panic!(1337));

    // Core panics
    test(None, |_old_val| core::panic!("Hello from panic: core"));
    test(None, |old_val| core::panic!(&format!("Hello from panic: {:?}", old_val)));
    test(None, |old_val| core::panic!("Hello from panic: {:?}", old_val));

    // Built-in panics; also make sure the message is right.
    test(Some("index out of bounds: the len is 3 but the index is 4"), |_old_val| {
        let _val = [0, 1, 2][4];
        loop {}
    });
    test(Some("attempt to divide by zero"), |_old_val| {
        let _val = 1 / 0;
        loop {}
    });

    test(Some("align_offset: align is not a power-of-two"), |_old_val| {
        (0usize as *const u8).align_offset(3);
        loop {}
    });

    // Assertion and debug assertion
    test(None, |_old_val| {
        assert!(false);
        loop {}
    });
    test(None, |_old_val| {
        debug_assert!(false);
        loop {}
    });

    eprintln!("Success!"); // Make sure we get this in stderr
}

fn test(expect_msg: Option<&str>, do_panic: impl FnOnce(usize) -> !) {
    // Reset test flags.
    DROPPED.with(|c| c.set(false));
    HOOK_CALLED.with(|c| c.set(false));

    // Cause and catch a panic.
    let res = catch_unwind(AssertUnwindSafe(|| {
        let _string = "LEAKED FROM CLOSURE".to_string();
        do_panic_counter(do_panic)
    }))
    .expect_err("do_panic() did not panic!");

    // See if we can extract the panic message.
    let msg = if let Some(s) = res.downcast_ref::<String>() {
        eprintln!("Caught panic message (String): {}", s);
        Some(s.as_str())
    } else if let Some(s) = res.downcast_ref::<&str>() {
        eprintln!("Caught panic message (&str): {}", s);
        Some(*s)
    } else {
        eprintln!("Failed to get caught panic message.");
        None
    };
    if let Some(expect_msg) = expect_msg {
        assert_eq!(expect_msg, msg.unwrap());
    }

    // Test flags.
    assert!(DROPPED.with(|c| c.get()));
    assert!(HOOK_CALLED.with(|c| c.get()));
}
