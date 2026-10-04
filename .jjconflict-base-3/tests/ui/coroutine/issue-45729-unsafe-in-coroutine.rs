#![feature(coroutines, stmt_expr_attributes)]

fn main() {
    let _ = #[coroutine]
    || {
        *(1 as *mut u32) = 42;
        //~^ ERROR dereference of raw pointer is unsafe
        yield;
    };
}
