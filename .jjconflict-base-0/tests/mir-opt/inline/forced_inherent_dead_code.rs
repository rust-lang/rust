// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ compile-flags: -Copt-level=0 -Clink-dead-code
#![feature(rustc_attrs)]

struct Foo {}

impl Foo {
    #[rustc_force_inline]
    pub fn callee_forced() {}
}

// EMIT_MIR forced_inherent_dead_code.caller.ForceInline.diff
pub fn caller() {
    Foo::callee_forced();
    // CHECK-LABEL: fn caller(
    // CHECK: (inlined Foo::callee_forced)
}

fn main() {
    caller();
}
