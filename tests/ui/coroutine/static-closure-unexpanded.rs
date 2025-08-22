// Tests that static closures are not stable in the parser grammar unless the
// coroutine feature is enabled.

#[cfg(any())]
fn foo() {
    let _ = static || {};
    //~^ ERROR coroutine syntax is experimental
}

fn main() {}
