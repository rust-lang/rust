// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ compile-flags: -Copt-level=0 --crate-type=lib
#![feature(rustc_attrs)]

struct Foo;

impl Foo {
    #[rustc_force_inline]
    fn bar() {}
}

trait Tr {
    fn bar();
}

impl Tr for Foo {
    fn bar() {}
}

// EMIT_MIR forced_inherent_ambiguous.caller.ForceInline.diff
fn caller() {
    Foo::bar();
    // CHECK-LABEL: fn caller(
    // CHECK: (inlined Foo::bar)
}
