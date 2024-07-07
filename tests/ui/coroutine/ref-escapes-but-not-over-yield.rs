#![feature(coroutines, stmt_expr_attributes)]

fn foo(x: &i32) {
    let a = &mut &3;
    let mut b = #[coroutine]
    move || {
        yield ();
        let b = 5;
        *a = &b;
        //~^ ERROR: borrowed data escapes outside of coroutine
    };
}

fn main() {}
