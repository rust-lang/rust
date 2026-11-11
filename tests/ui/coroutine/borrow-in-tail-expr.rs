//@ run-pass

#![feature(coroutines, stmt_expr_attributes)]

fn main() {
    let _a = #[coroutine] || {
        yield;
        let a = String::new();
        a.len()
    };
}
