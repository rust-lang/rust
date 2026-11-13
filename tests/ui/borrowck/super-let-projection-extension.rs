//! Demonstrates a case where `{ super let x = temp(); &x }` =/= `&temp()`, a variant of which is
//! observable on stable Rust via the `pin!` macro: `pin!($EXPR)` and `&mut $EXPR` may use different
//! scopes for their temporaries.

#![feature(super_let)]

use std::pin::pin;

fn temp() {}

fn main() {
    // This is fine, since the temporary is extended to the end of the block:
    let a = &*&temp();
    a;
    let b = &mut *&mut temp();
    b;

    // The temporary is dropped at the end of the outer `let` initializer:
    let c = &*{ super let x = temp(); &x };
    //~^ ERROR `x` does not live long enough
    c;
    let d = &mut *pin!(temp());
    //~^ ERROR temporary value dropped while borrowed
    d;
}
