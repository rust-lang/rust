#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::ops::Coroutine;
use std::pin::Pin;

fn main() {
    let _b = {
        let a = 3;
        Pin::new(&mut #[coroutine] || (&a).yield).resume(())
        //~^ ERROR: `a` does not live long enough
    };

    let _b = {
        let a = 3;
        #[coroutine] || {
            (&a).yield
            //~^ ERROR: `a` does not live long enough
        }
    };
}
