//@ run-pass

// Based on addassign-yield.rs, but with drop tracking enabled. Originally we did not implement
// the fake_read callback on ExprUseVisitor which caused this case to break.

#![feature(coroutines, stmt_expr_attributes)]

fn foo() {
    let _y = #[coroutine] static || {
        let x = &mut 0;
        *{
            yield;
            x
        } += match String::new() {
            _ => 0,
        };
    };

    // Please don't ever actually write something like this
    let _z = #[coroutine] static || {
        let x = &mut 0;
        *{
            let inner = &mut 1;
            *{
                yield ();
                inner
            } += match String::new() {
                _ => 1,
            };
            yield;
            x
        } += match String::new() {
            _ => 2,
        };
    };
}

fn main() {
    foo()
}
