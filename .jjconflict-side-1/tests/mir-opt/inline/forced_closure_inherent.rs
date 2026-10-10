// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ compile-flags: -Copt-level=0 --crate-type=lib
#![feature(rustc_attrs)]

struct Foo {}

impl Foo {
    #[rustc_force_inline]
    pub fn callee_forced() {}
}

// EMIT_MIR forced_closure_inherent.caller-{closure#0}.ForceInline.diff
pub fn caller() {
    (|| {
        Foo::callee_forced();
        // CHECK-LABEL: fn caller::{closure#0}(
        // CHECK: (inlined Foo::callee_forced)
    })();
}
