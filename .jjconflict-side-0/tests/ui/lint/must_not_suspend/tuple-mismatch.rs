#![feature(coroutines, stmt_expr_attributes)]

fn main() {
    let _coroutine = #[coroutine]
    || {
        yield ((), ((), ()));
        yield ((), ());
        //~^ ERROR mismatched types
    };
}
