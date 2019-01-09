// run-pass
#![allow(unused_unsafe)]

#![feature(generators)]

fn main() {
    unsafe {
        static move || {
            // Tests that the generator transformation finds out that `a` is not live
            // during the yield expression. Type checking will also compute liveness
            // and it should also find out that `a` is not live.
            // The compiler will panic if the generator transformation finds that
            // `a` is live and type checking finds it dead.
            let a = {
                yield ();
                4i32
            };
            &a;
        };
    }
}
