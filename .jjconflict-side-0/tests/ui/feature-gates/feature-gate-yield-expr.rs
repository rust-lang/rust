//@ edition: 2024
#![feature(stmt_expr_attributes)]

fn main() {
    yield (); //~ ERROR yield syntax is experimental
    //~^ ERROR yield syntax is experimental
    //~^^ ERROR `yield` can only be used in `#[coroutine]` closures, or `gen` blocks
    //~^^^ ERROR yield expression outside of coroutine literal
}
