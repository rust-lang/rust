//! Test using `#[splat]` on tuple arguments of pointers to simple functions.
//! Currently ICEs, but if we fix it, we'll want to know and update this test to pass.

//@ failure-status: 101

//@ normalize-stderr: ".*error:.*compiler/([^:]+):\d{1,}:\d{1,}:(.*)" -> "error: compiler/$1:LL:CC:$2"
//@ normalize-stderr: "thread.*panicked at .*compiler.*" -> ""
//@ normalize-stderr: "note: rustc.*running on.*" -> "note: rustc {version} running on {platform}"
//@ normalize-stderr: "note: compiler flags.*\n\n" -> ""
//@ normalize-stderr: " +\d{1,}: .*\n" -> ""
//@ normalize-stderr: " + at .*\n" -> ""
//@ normalize-stderr: ".*omitted \d{1,} frames?.*\n" -> ""
//@ normalize-stderr: ".*note: Some details are omitted.*\n" -> ""
//@ normalize-stderr: ".*--> .*/splat-fn-ptr-tuple.rs:\d{1,}:\d{1,}.*\n" -> ""

#![allow(incomplete_features)]
#![feature(splat)]

fn tuple_args(#[splat] (_a, _b): (u32, i8)) {}

fn splat_non_terminal_arg(#[splat] (_a, _b): (u32, i8), _c: f64) {}

fn main() {
    // FIXME(splat): not currently supported, can be supported when we no longer require a DefId in
    // MIR lowering
    // FIXME(rustfmt): the attribute gets deleted by rustfmt
    #[rustfmt::skip]
    let fn_ptr: fn(#[splat] (u32, i8)) = tuple_args;
    fn_ptr(1, 2); //~ ERROR splatted FnPtr side-tables are not yet implemented
    // The ICE means that code after this line is not fully checked
    fn_ptr(1u32, 2i8);

    // FIXME(splat): should splatted functions be callable with tupled and un-tupled arguments?
    // Add a tupled test for each call if they are.
    //fn_ptr((1, 2)); // ERROR this splatted function takes 2 arguments, but 1 was provided

    #[rustfmt::skip]
    let fn_ptr: fn(#[splat] (u32, i8), f64) = splat_non_terminal_arg;
    fn_ptr(1, 2, 3.5);
    fn_ptr(1u32, 2i8, 3.5f64);

    // Bug #158603 regression test
    #[rustfmt::skip]
    let x: fn(#[splat] (i32,)) = None.unwrap();
    x(1);
}
