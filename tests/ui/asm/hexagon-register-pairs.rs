//@ add-core-stubs
//@ compile-flags: --target hexagon-unknown-linux-musl
//@ needs-llvm-components: hexagon

#![feature(no_core, asm_experimental_arch)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::*;

fn test_register_spans() {
    unsafe {
        // These are valid Hexagon register span notations, not labels
        // Should NOT trigger the named labels lint

        // General register pairs
        asm!("r1:0 = memd(r29+#0)", lateout("r0") _, lateout("r1") _);
        asm!("r3:2 = combine(#1, #0)", lateout("r2") _, lateout("r3") _);
        asm!("r15:14 = memd(r30+#8)", lateout("r14") _, lateout("r15") _);
        asm!("memd(r29+#0) = r5:4", in("r4") 0u32, in("r5") 0u32);

        // Vector register pairs (hypothetical since we can't actually use them without HVX)
        asm!("nop // V5:4 = vadd(V3:2, V1:0)"); // Comment form to avoid actual instruction issues
        asm!("nop // V7:6.w = vadd(V5:4.w, V3:2.w)"); // With type suffixes

        // Predicate register pairs
        asm!("nop // p1:0 = vcmpb.eq(V1:0, V3:2)"); // Comment form

        // Mixed with actual labels should still trigger for the labels
        asm!("label1: r7:6 = combine(#2, #3)"); //~ ERROR avoid using named labels

        // Regular labels should still trigger
        asm!("hexagon_label: nop"); //~ ERROR avoid using named labels
    }
}
