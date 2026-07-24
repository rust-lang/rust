//! Test using `#[arg_splat]` on tuple arguments of generic function constants.
//! Currently ICEs (#158603), but if we fix it, we'll want to know and update this test to pass.

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
#![feature(arg_splat, tuple_trait)]

use std::marker::Tuple;

fn f<Args: Tuple>(#[arg_splat] args: Args) {}

fn main() {
    // FIXME(splat): not currently supported, can be supported when we no longer require a DefId in
    // MIR lowering
    // FIXME(rustfmt): the attribute gets deleted by rustfmt
    #[rustfmt::skip]
    const F2: fn(#[arg_splat] (u8, u32)) = f::<(u8, u32)>;
    const R2: () = F2(1, 2); //~ ERROR splatted FnPtr side-tables are not yet implemented

    #[rustfmt::skip]
    const F1: fn(#[arg_splat] ((u8, u32),)) = f::<((u8, u32),)>;
    const R1: () = F1((1, 2)); //~ ERROR splatted FnPtr side-tables are not yet implemented
}
