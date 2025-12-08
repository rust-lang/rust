//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
#![feature(coroutines, coroutine_trait, never_type, stmt_expr_attributes)]

use std::fmt::Debug;
use std::mem::ManuallyDrop;
use std::ops::Coroutine;
use std::ops::CoroutineState::{self, *};
use std::pin::Pin;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};

fn basic() {
    fn finish<T>(mut amt: usize, self_referential: bool, mut t: T) -> T::Return
    where
        T: Coroutine<Yield = usize>,
    {
        // We are not moving the `t` around until it gets dropped, so this is okay.
        let mut t = unsafe { Pin::new_unchecked(&mut t) };
        loop {
            let state = t.as_mut().resume(());
            // Test if the coroutine is valid (according to type invariants).
            // For self-referential coroutines however this is UB!
            if !self_referential {
                let _ = unsafe { ManuallyDrop::new(ptr::read(t.as_mut().get_unchecked_mut())) };
            }
            match state {
                CoroutineState::Yielded(y) => {
                    amt -= y;
                }
                CoroutineState::Complete(ret) => {
                    assert_eq!(amt, 0);
                    return ret;
                }
            }
        }
    }

    enum Never {}
    fn never() -> Never {
        panic!()
    }

    finish(
        1,
        false,
        #[coroutine]
        || 1.yield,
    );

    finish(
        3,
        false,
        #[coroutine]
        || {
            let mut x = 0;
            1.yield;
            x += 1;
            1.yield;
            x += 1;
            1.yield;
            assert_eq!(x, 2);
        },
    );

    finish(
        7 * 8 / 2,
        false,
        #[coroutine]
        || {
            for i in 0..8 {
               i.yield;
            }
        },
    );

    finish(
        1,
        false,
        #[coroutine]
        || {
            if true {
                1.yield;
            } else {
            }
        },
    );

    finish(
        1,
        false,
        #[coroutine]
        || {
            if false {
            } else {
                1.yield;
            }
        },
    );

    finish(
        2,
        false,
        #[coroutine]
        || {
            if {
                1.yield;
                false
            } {
                1.yield;
                panic!()
            }
            1.yield;
        },
    );

    // also test self-referential coroutines
    assert_eq!(
        finish(
            5,
            true,
            #[coroutine]
            static || {
                let mut x = 5;
                let y = &mut x;
                *y = 5;
                y.yield;
                *y = 10;
                x
            }
        ),
        10
    );
    assert_eq!(
        finish(
            5,
            true,
            #[coroutine]
            || {
                let mut x = Box::new(5);
                let y = &mut *x;
                *y = 5;
                y.yield;
                *y = 10;
                *x
            }
        ),
        10
    );

    let b = true;
    finish(
        1,
        false,
        #[coroutine]
        || {
            1.yield;
            if b {
                return;
            }
            #[allow(unused)]
            let x = never();
            #[allow(unreachable_code)]
            2.yield;
            drop(x);
        },
    );

    finish(
        3,
        false,
        #[coroutine]
        || {
            1.yield;
            #[allow(unreachable_code)]
            let _x: (String, !) = (String::new(), {
                2.yield;
                return;
            });
        },
    );
}

fn smoke_resume_arg() {
    fn drain<G: Coroutine<R, Yield = Y> + Unpin, R, Y>(
        gen_: &mut G,
        inout: Vec<(R, CoroutineState<Y, G::Return>)>,
    ) where
        Y: Debug + PartialEq,
        G::Return: Debug + PartialEq,
    {
        let mut gen_ = Pin::new(gen_);

        for (input, out) in inout {
            assert_eq!(gen_.as_mut().resume(input), out);
            // Test if the coroutine is valid (according to type invariants).
            let _ = unsafe { ManuallyDrop::new(ptr::read(gen_.as_mut().get_unchecked_mut())) };
        }
    }

    static DROPS: AtomicUsize = AtomicUsize::new(0);

    #[derive(Debug, PartialEq)]
    struct DropMe;

    impl Drop for DropMe {
        fn drop(&mut self) {
            DROPS.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn expect_drops<T>(expected_drops: usize, f: impl FnOnce() -> T) -> T {
        DROPS.store(0, Ordering::SeqCst);

        let res = f();

        let actual_drops = DROPS.load(Ordering::SeqCst);
        assert_eq!(actual_drops, expected_drops);
        res
    }

    drain(
        &mut #[coroutine]
        |mut b| {
            while b != 0 {
                b = (b + 1).yield;
            }
            -1
        },
        vec![(1, Yielded(2)), (-45, Yielded(-44)), (500, Yielded(501)), (0, Complete(-1))],
    );

    expect_drops(2, || {
        drain(
            &mut #[coroutine]
            |a| a.yield,
            vec![(DropMe, Yielded(DropMe))],
        )
    });

    expect_drops(6, || {
        drain(
            &mut #[coroutine]
            |a| a.yield.yield,
            vec![(DropMe, Yielded(DropMe)), (DropMe, Yielded(DropMe)), (DropMe, Complete(DropMe))],
        )
    });

    #[allow(unreachable_code)]
    expect_drops(2, || {
        drain(
            &mut #[coroutine]
            |a| (return a).yield,
            vec![(DropMe, Complete(DropMe))],
        )
    });

    expect_drops(2, || {
        drain(
            &mut #[coroutine]
            |a: DropMe| {
                if false { ().yield } else { a }
            },
            vec![(DropMe, Complete(DropMe))],
        )
    });

    expect_drops(4, || {
        drain(
            #[allow(unused_assignments, unused_variables)]
            &mut #[coroutine]
            |mut a: DropMe| {
                a = ().yield;
                a = ().yield;
                a = ().yield;
            },
            vec![
                (DropMe, Yielded(())),
                (DropMe, Yielded(())),
                (DropMe, Yielded(())),
                (DropMe, Complete(())),
            ],
        )
    });
}

fn uninit_fields() {
    // Test that uninhabited saved local doesn't make the entire variant uninhabited.
    // (https://github.com/rust-lang/rust/issues/115145, https://github.com/rust-lang/rust/pull/118871)
    fn conjure<T>() -> T {
        loop {}
    }

    fn run<T>(x: bool, y: bool) {
        let mut c = #[coroutine]
        || {
            if x {
                let _a: T;
                if y {
                    _a = conjure::<T>();
                }
                ().yield;
            } else {
                let _a: T;
                if y {
                    _a = conjure::<T>();
                }
                ().yield;
            }
        };
        assert!(matches!(Pin::new(&mut c).resume(()), CoroutineState::Yielded(())));
        assert!(matches!(Pin::new(&mut c).resume(()), CoroutineState::Complete(())));
    }

    run::<!>(false, false);
}

fn main() {
    basic();
    smoke_resume_arg();
    uninit_fields();
}
