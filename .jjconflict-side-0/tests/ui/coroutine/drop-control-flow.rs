//@ build-pass

// A test to ensure coroutines capture values that were conditionally dropped,
// and also that values that are dropped along all paths to a yield do not get
// included in the coroutine type.

#![feature(coroutines, negative_impls, stmt_expr_attributes)]
#![allow(unused_assignments, dead_code)]

struct Ptr;
impl<'a> Drop for Ptr {
    fn drop(&mut self) {}
}

struct NonSend;
impl !Send for NonSend {}

fn assert_send<T: Send>(_: T) {}

// This test case is reduced from tests/ui/drop/dynamic-drop-async.rs
fn one_armed_if(arg: bool) {
    let _ = #[coroutine] || {
        let arr = [Ptr];
        if arg {
            drop(arr);
        }
        yield;
    };
}

fn two_armed_if(arg: bool) {
    assert_send(#[coroutine] || {
        let arr = [Ptr];
        if arg {
            drop(arr);
        } else {
            drop(arr);
        }
        yield;
    })
}

fn if_let(arg: Option<i32>) {
    let _ = #[coroutine] || {
        let arr = [Ptr];
        if let Some(_) = arg {
            drop(arr);
        }
        yield;
    };
}

fn init_in_if(arg: bool) {
    assert_send(#[coroutine] || {
        let mut x = NonSend;
        drop(x);
        if arg {
            x = NonSend;
        } else {
            yield;
        }
    })
}

fn init_in_match_arm(arg: Option<i32>) {
    assert_send(#[coroutine] || {
        let mut x = NonSend;
        drop(x);
        match arg {
            Some(_) => x = NonSend,
            None => yield,
        }
    })
}

fn reinit() {
    let _ = #[coroutine] || {
        let mut arr = [Ptr];
        drop(arr);
        arr = [Ptr];
        yield;
    };
}

fn loop_uninit() {
    let _ = #[coroutine] || {
        let mut arr = [Ptr];
        let mut count = 0;
        drop(arr);
        while count < 3 {
            yield;
            arr = [Ptr];
            count += 1;
        }
    };
}

fn nested_loop() {
    let _ = #[coroutine] || {
        let mut arr = [Ptr];
        let mut count = 0;
        drop(arr);
        while count < 3 {
            for _ in 0..3 {
                yield;
            }
            arr = [Ptr];
            count += 1;
        }
    };
}

fn loop_continue(b: bool) {
    let _ = #[coroutine] || {
        let mut arr = [Ptr];
        let mut count = 0;
        drop(arr);
        while count < 3 {
            count += 1;
            yield;
            if b {
                arr = [Ptr];
                continue;
            }
        }
    };
}

fn main() {
    one_armed_if(true);
    if_let(Some(41));
    init_in_if(true);
    init_in_match_arm(Some(41));
    reinit();
    loop_uninit();
    nested_loop();
    loop_continue(true);
}
