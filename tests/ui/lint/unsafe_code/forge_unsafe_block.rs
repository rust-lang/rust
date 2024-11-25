//@ check-pass
//@ proc-macro: forge_unsafe_block.rs

#[macro_use]
extern crate forge_unsafe_block;

unsafe fn foo() {}

#[forbid(unsafe_code)]
fn main() {
    // `forbid` doesn't work for non-user-provided unsafe blocks.
    // see `UnsafeCode::check_expr`.
    forge_unsafe_block! {
        foo();
    }
}
