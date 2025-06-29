//@ compile-flags: -Zverbose-internals

#![feature(coroutines, stmt_expr_attributes)]

fn main() {
    let x = #[coroutine] || {};
    //~^ NOTE `x` has type `{main::{closure:coroutine#0} upvar_tys=?3t resume_ty=() yield_ty=?4t return_ty=() witness=?5t}`
    let () = x();
    //~^ ERROR expected function, found `{main::{closure:coroutine#0} upvar_tys=?3t resume_ty=() yield_ty=?4t return_ty=() witness=?5t}`
    //~| NOTE call expression requires function
}
