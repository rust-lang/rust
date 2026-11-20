// Make sure we don't suggest compatible variants cross macro context. (issue #142359)
//@ aux-crate:ext=suggest-compatible-variants-macro-issue-142359.rs

extern crate ext;

use std::ops::ControlFlow;

fn main() {
    let x: Result<i32, i32> = Err(1);

    let _ = match x {
        Err(r) => ControlFlow::Break(r),
        Ok(r) => { ext::unit!() } //~ ERROR `match` arms have incompatible types [E0308]

    };
}
