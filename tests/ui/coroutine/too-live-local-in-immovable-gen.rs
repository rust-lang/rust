//@ run-pass
#![allow(unused_unsafe)]

#![feature(coroutines)]

fn main() {
    unsafe {
        #[coroutine] static move || { //~ WARN unused coroutine that must be used
            // Tests that the coroutine transformation finds out that `a` is not live
            // during the yield expression. Type checking will also compute liveness
            // and it should also find out that `a` is not live.
            // The compiler will panic if the coroutine transformation finds that
            // `a` is live and type checking finds it dead.
            let a = {
                yield ();
                4i32
            };
            let _ = &a;
        };
    }
}
