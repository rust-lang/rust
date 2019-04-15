use std::panic::catch_unwind;
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

fn do_panic_counter() {
    // If this gets leaked, it will be easy to spot
    // in Miri's leak report
    let _string = "LEAKED FROM do_panic_counter".to_string();

    // When we panic, this should get dropped during unwinding
    let _drop_tester = DropTester;

    // Check for bugs in Miri's panic implementation.
    // If do_panic_counter() somehow gets called more than once,
    // we'll generate a different panic message
    let old_val = MY_COUNTER.with(|c| {
        let val = c.get();
        c.set(val + 1);
        val
    });
    panic!(format!("Hello from panic: {:?}", old_val));
}

fn main() {
    std::panic::set_hook(Box::new(|_panic_info| {
        HOOK_CALLED.with(|h| h.set(true));
    }));
    let res = catch_unwind(|| {
        let _string = "LEAKED FROM CLOSURE".to_string();
        do_panic_counter()
    });
    let expected: Box<String> = Box::new("Hello from panic: 0".to_string());
    let actual = res.expect_err("do_panic() did not panic!")
        .downcast::<String>().expect("Failed to cast to string!");
        
    assert_eq!(expected, actual);
    DROPPED.with(|c| {
        // This should have been set to 'true' by DropTester
        assert!(c.get());
    });

    HOOK_CALLED.with(|h| {
        assert!(h.get());
    });
}

