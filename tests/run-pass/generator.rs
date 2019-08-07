#![feature(generators, generator_trait)]

use std::ops::{GeneratorState, Generator};
use std::pin::Pin;

fn finish<T>(mut amt: usize, mut t: T) -> T::Return
    where T: Generator<Yield = usize>
{
    // We are not moving the `t` around until it gets dropped, so this is okay.
    let mut t = unsafe { Pin::new_unchecked(&mut t) };
    loop {
        match t.as_mut().resume() {
            GeneratorState::Yielded(y) => amt -= y,
            GeneratorState::Complete(ret) => {
                assert_eq!(amt, 0);
                return ret
            }
        }
    }
}

enum Never {}
fn never() -> Never {
    panic!()
}

fn main() {
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
    finish(7*8/2, || {
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
        if { yield 1; false } {
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
        if b { return; }
        #[allow(unused)]
        let x = never();
        yield 2;
        drop(x);
    });
}
