// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ compile-flags: -Copt-level=0 --crate-type=lib
//@ edition: 2021
#![feature(rustc_attrs)]

struct Foo {}

impl Foo {
    #[rustc_force_inline]
    pub fn callee_forced() {}
}

// EMIT_MIR forced_inherent_async.caller.ForceInline.diff
async fn caller() {
    Foo::callee_forced();
    // CHECK-LABEL: fn caller(
    // CHECK: (inlined Foo::callee_forced)
}
