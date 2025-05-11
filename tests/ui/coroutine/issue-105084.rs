//@compile-flags: --diagnostic-width=300
#![feature(coroutines)]
#![feature(coroutine_clone)]
#![feature(coroutine_trait)]
#![feature(rustc_attrs, stmt_expr_attributes, liballoc_internals)]

use std::ops::Coroutine;
use std::pin::Pin;

fn copy<T: Copy>(x: T) -> T {
    x
}

fn main() {
    let mut g = #[coroutine]
    || {
        // This is desuraged as 4 stages:
        // - allocate a `*mut u8` with `exchange_malloc`;
        // - create a Box that is ignored for trait computations;
        // - compute fields (and yields);
        // - assign to `t`.
        let t = std::boxed::box_new((5, yield));
        drop(t);
    };

    // Allocate the temporary box.
    Pin::new(&mut g).resume(());

    // The temporary box is in coroutine locals.
    // As it is not taken into account for trait computation,
    // the coroutine is `Copy`.
    let mut h = copy(g);
    //~^ ERROR the trait bound `Box<(i32, ())>: Copy` is not satisfied in

    // We now have 2 boxes with the same backing allocation:
    // one inside `g` and one inside `h`.
    // Proceed and drop `t` in `g`.
    Pin::new(&mut g).resume(());
    //~^ ERROR borrow of moved value: `g`

    // Proceed and drop `t` in `h` -> double free!
    Pin::new(&mut h).resume(());
}
