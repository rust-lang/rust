//@ run-pass
// Regression test for broken MIR error (#61442)
// Due to the two possible evaluation orders for
// a '+=' expression (depending on whether or not the 'AddAssign' trait
// is being used), we were failing to account for all types that might
// possibly be live across a yield point.

#![feature(coroutines, stmt_expr_attributes)]

fn foo() {
    let _x = #[coroutine] static || {
        let mut s = String::new();
        s += { yield; "" };
    };

    let _y = #[coroutine] static || {
        let x = &mut 0;
        *{ yield; x } += match String::new() { _ => 0 };
    };

    // Please don't ever actually write something like this
    let _z = #[coroutine] static || {
        let x = &mut 0;
        *{
            let inner = &mut 1;
            *{ yield (); inner } += match String::new() { _ => 1};
            yield;
            x
        } += match String::new() { _ => 2 };
    };
}

fn main() {
    foo()
}
