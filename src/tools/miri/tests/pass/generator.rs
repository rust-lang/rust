//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
#![feature(generators, generator_trait, never_type)]

use std::fmt::Debug;
use std::mem::ManuallyDrop;
use std::ops::{
    Generator,
    GeneratorState::{self, *},
};
use std::pin::Pin;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};

fn basic() {
    fn finish<T>(mut amt: usize, mut t: T) -> T::Return
    where
        T: Generator<Yield = usize>,
    {
        // We are not moving the `t` around until it gets dropped, so this is okay.
        let mut t = unsafe { Pin::new_unchecked(&mut t) };
        loop {
            let state = t.as_mut().resume(());
            // Test if the generator is valid (according to type invariants).
            let _ = unsafe { ManuallyDrop::new(ptr::read(t.as_mut().get_unchecked_mut())) };
            match state {
                GeneratorState::Yielded(y) => {
                    amt -= y;
                }
                GeneratorState::Complete(ret) => {
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

    finish(1, || yield 1);

    finish(3, || {
        let mut x = 0;
        yield 1;
        x += 1;
        yield 1;
        x += 1;
        yield 1;
        assert_eq!(x, 2);
    });

    finish(7 * 8 / 2, || {
        for i in 0..8 {
            yield i;
        }
    });

    finish(1, || {
        if true {
            yield 1;
        } else {
        }
    });

    finish(1, || {
        if false {
        } else {
            yield 1;
        }
    });

    finish(2, || {
        if {
            yield 1;
            false
        } {
            yield 1;
            panic!()
        }
        yield 1;
    });

    // also test a self-referential generator
    assert_eq!(
        finish(5, || {
            let mut x = Box::new(5);
            let y = &mut *x;
            *y = 5;
            yield *y;
            *y = 10;
            *x
        }),
        10
    );

    let b = true;
    finish(1, || {
        yield 1;
        if b {
            return;
        }
        #[allow(unused)]
        let x = never();
        #[allow(unreachable_code)]
        yield 2;
        drop(x);
    });

    finish(3, || {
        yield 1;
        #[allow(unreachable_code)]
        let _x: (String, !) = (String::new(), {
            yield 2;
            return;
        });
    });
}

fn smoke_resume_arg() {
    fn drain<G: Generator<R, Yield = Y> + Unpin, R, Y>(
        gen: &mut G,
        inout: Vec<(R, GeneratorState<Y, G::Return>)>,
    ) where
        Y: Debug + PartialEq,
        G::Return: Debug + PartialEq,
    {
        let mut gen = Pin::new(gen);

        for (input, out) in inout {
            assert_eq!(gen.as_mut().resume(input), out);
            // Test if the generator is valid (according to type invariants).
            let _ = unsafe { ManuallyDrop::new(ptr::read(gen.as_mut().get_unchecked_mut())) };
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
        &mut |mut b| {
            while b != 0 {
                b = yield (b + 1);
            }
            -1
        },
        vec![(1, Yielded(2)), (-45, Yielded(-44)), (500, Yielded(501)), (0, Complete(-1))],
    );

    expect_drops(2, || drain(&mut |a| yield a, vec![(DropMe, Yielded(DropMe))]));

    expect_drops(6, || {
        drain(
            &mut |a| yield yield a,
            vec![(DropMe, Yielded(DropMe)), (DropMe, Yielded(DropMe)), (DropMe, Complete(DropMe))],
        )
    });

    #[allow(unreachable_code)]
    expect_drops(2, || drain(&mut |a| yield return a, vec![(DropMe, Complete(DropMe))]));

    expect_drops(2, || {
        drain(
            &mut |a: DropMe| {
                if false { yield () } else { a }
            },
            vec![(DropMe, Complete(DropMe))],
        )
    });

    expect_drops(4, || {
        drain(
            #[allow(unused_assignments, unused_variables)]
            &mut |mut a: DropMe| {
                a = yield;
                a = yield;
                a = yield;
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

fn main() {
    basic();
    smoke_resume_arg();
}
