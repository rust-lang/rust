// Edition 2024 lint for change in drop order at tail expression
// This lint is to capture potential borrow-checking errors
// due to implementation of RFC 3606 <https://github.com/rust-lang/rfcs/pull/3606>
//@ edition: 2021

#![deny(tail_expr_drop_order)] //~ NOTE: the lint level is defined here

fn should_lint_with_potential_borrowck_err() {
    let _ = { String::new().as_str() }.len();
    //~^ ERROR: relative drop order changing
    //~| WARN: this changes meaning in Rust 2024
    //~| NOTE: this temporary value will be dropped at the end of the block
    //~| NOTE: borrow later used by call
    //~| NOTE: for more information, see
}

fn should_lint_with_unsafe_block() {
    fn f(_: usize) {}
    f(unsafe { String::new().as_str() }.len());
    //~^ ERROR: relative drop order changing
    //~| WARN: this changes meaning in Rust 2024
    //~| NOTE: this temporary value will be dropped at the end of the block
    //~| NOTE: borrow later used by call
    //~| NOTE: for more information, see
}

#[rustfmt::skip]
fn should_lint_with_big_block() {
    fn f<T>(_: T) {}
    f({
        &mut || 0
        //~^ ERROR: relative drop order changing
        //~| WARN: this changes meaning in Rust 2024
        //~| NOTE: this temporary value will be dropped at the end of the block
        //~| NOTE: borrow later used here
        //~| NOTE: for more information, see
    })
}

fn another_temp_that_is_copy_in_arg() {
    fn f() {}
    fn g(_: &()) {}
    g({ &f() });
    //~^ ERROR: relative drop order changing
    //~| WARN: this changes meaning in Rust 2024
    //~| NOTE: this temporary value will be dropped at the end of the block
    //~| NOTE: borrow later used by call
    //~| NOTE: for more information, see
}

fn main() {}
