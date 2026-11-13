#![feature(coroutines, stmt_expr_attributes)]

fn foo(x: &i32) {
    let mut a = &3;
    let b = #[coroutine]
    move || {
        yield ();
        let b = 5;
        a = &b;
        //~^ ERROR: borrowed data escapes outside of coroutine
    };
}

fn main() {}
