// revisions: no_drop_tracking drop_tracking drop_tracking_mir
// [drop_tracking] compile-flags: -Zdrop-tracking
// [drop_tracking_mir] compile-flags: -Zdrop-tracking-mir
// [no_drop_tracking] known-bug: #105084
// [no_drop_tracking] check-pass
// [drop_tracking] known-bug: #105084
// [drop_tracking] check-pass

#![feature(generators)]
#![feature(generator_clone)]
#![feature(generator_trait)]
#![feature(rustc_attrs, stmt_expr_attributes)]

use std::ops::Generator;
use std::pin::Pin;

fn copy<T: Copy>(x: T) -> T {
    x
}

fn main() {
    let mut g = || {
        // This is desuraged as 4 stages:
        // - allocate a `*mut u8` with `exchange_malloc`;
        // - create a Box that is ignored for trait computations;
        // - compute fields (and yields);
        // - assign to `t`.
        let t = #[rustc_box]
        Box::new((5, yield));
        drop(t);
    };

    // Allocate the temporary box.
    Pin::new(&mut g).resume(());

    // The temporary box is in generator locals.
    // As it is not taken into account for trait computation,
    // the generator is `Copy`.
    let mut h = copy(g);
    //[drop_tracking_mir]~^ ERROR the trait bound `Box<(i32, ())>: Copy` is not satisfied in

    // We now have 2 boxes with the same backing allocation:
    // one inside `g` and one inside `h`.
    // Proceed and drop `t` in `g`.
    Pin::new(&mut g).resume(());
    //[drop_tracking_mir]~^ ERROR borrow of moved value: `g`

    // Proceed and drop `t` in `h` -> double free!
    Pin::new(&mut h).resume(());
}
