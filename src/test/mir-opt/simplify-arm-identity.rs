// Checks that `SimplifyArmIdentity` is not applied if enums have incompatible layouts.
// Regression test for issue #66856.
//
// compile-flags: -Zmir-opt-level=3
// EMIT_MIR_FOR_EACH_BIT_WIDTH

// This pass is broken since deaggregation changed
// ignore-test

enum Src {
    Foo(u8),
    Bar,
}

enum Dst {
    Foo(u8),
}

// EMIT_MIR simplify_arm_identity.main.SimplifyArmIdentity.diff
fn main() {
    let e: Src = Src::Foo(0);
    let _: Dst = match e {
        Src::Foo(x) => Dst::Foo(x),
        Src::Bar => Dst::Foo(0),
    };
}
