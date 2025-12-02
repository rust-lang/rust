//@ edition:2018
//@ check-pass
#![feature(coroutines, stmt_expr_attributes)]
#![warn(unused)]
#![allow(unreachable_code)]

pub fn unintentional_copy_one() {
    let mut last = None;
    let mut f = move |s| {
        last = Some(s); //~  WARN value captured by `last` is never read
                        //~| WARN value assigned to `last` is never read
    };
    f("a");
    f("b");
    f("c");
    dbg!(last.unwrap());
}

pub fn unintentional_copy_two() {
    let mut sum = 0;
    (1..10).for_each(move |x| {
        sum += x;
        //~^ WARN value captured by `sum` is never read
        //~| WARN value assigned to `sum` is never read
    });
    dbg!(sum);
}

pub fn f() {
    let mut c = 0;

    // Captured by value, but variable is dead on entry.
    let _ = move || {
        c = 1; //~ WARN value captured by `c` is never read
        println!("{}", c);
    };
    let _ = async move {
        c = 1; //~ WARN value captured by `c` is never read
        println!("{}", c);
    };

    // Read and written to, but never actually used.
    let _ = move || {
        c += 1;
        //~^ WARN value captured by `c` is never read
        //~|  WARN value assigned to `c` is never read
    };
    let _ = async move {
        c += 1;
        //~^ WARN value captured by `c` is never read
        //~|  WARN value assigned to `c` is never read
    };

    let _ = move || {
        println!("{}", c);
        // Value is read by closure itself on later invocations.
        c += 1;
    };
    let b = Box::new(42);
    let _ = move || {
        println!("{}", c);
        // Never read because this is FnOnce closure.
        c += 1; //~  WARN value assigned to `c` is never read
        drop(b);
    };
    let _ = async move {
        println!("{}", c);
        // Never read because this is a coroutine.
        c += 1; //~  WARN value assigned to `c` is never read
    };
}

pub fn nested() {
    let mut d = None;
    let mut e = None;
    let _ = || {
        let _ = || {
            d = Some("d1"); //~ WARN value assigned to `d` is never read
            d = Some("d2");
        };
        let _ = move || {
            e = Some("e1"); //~  WARN value captured by `e` is never read
                            //~| WARN value assigned to `e` is never read
            e = Some("e2"); //~  WARN value assigned to `e` is never read
        };
    };
}

pub fn g<T: Default>(mut v: T) {
    let _ = |r| {
        if r {
            v = T::default();
            //~^ WARN value assigned to `v` is never read
        } else {
            drop(v);
        }
    };
}

pub fn h<T: Copy + Default + std::fmt::Debug>() {
    let mut z = T::default();
    let _ = move |b| {
        loop {
            if b {
                z = T::default(); //~  WARN value captured by `z` is never read
                                  //~| WARN value assigned to `z` is never read
            } else {
                return;
            }
        }
        dbg!(z);
    };
}

async fn yield_now() {
    todo!();
}

pub fn async_coroutine() {
    let mut state: u32 = 0;

    let _ = async {
        state = 1;
        yield_now().await;
        state = 2;
        yield_now().await;
        state = 3;
    };

    let _ = async move {
        state = 4;  //~  WARN value assigned to `state` is never read
                    //~| WARN value captured by `state` is never read
        yield_now().await;
        state = 5;  //~ WARN value assigned to `state` is never read
    };
}

pub fn coroutine() {
    let mut s: u32 = 0;
    let _ = #[coroutine] |_| {
        s = 0;
        yield ();
        s = 1; //~ WARN value assigned to `s` is never read
        yield (s = 2);
        s = yield (); //~ WARN value assigned to `s` is never read
        s = 3;
    };
}

pub fn panics() {
    use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

    let mut panic = true;

    // `a` can be called again, even if it has panicked at an earlier run.
    let mut a = || {
        if panic {
            panic = false;
            resume_unwind(Box::new(()))
        }
    };

    catch_unwind(AssertUnwindSafe(|| a())).ok();
    a();
}

fn main() {}
