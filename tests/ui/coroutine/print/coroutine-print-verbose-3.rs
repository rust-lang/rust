//@ compile-flags: -Zverbose-internals

#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

fn main() {
    let x = "Type mismatch test";
    let coroutine: () = #[coroutine]
    || {
        //~^ ERROR mismatched types
        yield 1i32;
        return x;
    };
}
