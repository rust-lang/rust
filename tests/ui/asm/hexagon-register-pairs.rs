//@ add-minicore
//@ compile-flags: --target hexagon-unknown-linux-musl -C target-feature=+hvx-length128b
//@ needs-llvm-components: hexagon
//@ ignore-backends: gcc

#![feature(no_core, asm_experimental_arch)]
#![crate_type = "lib"]
#![no_core]

//~? WARN unstable feature specified for `-Ctarget-feature`: `hvx-length128b`

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

        // These patterns look like register spans but test different edge cases
        // All should NOT trigger the lint as they match valid hexagon register syntax patterns
        asm!("V5:4 = vaddw(v1:0, v1:0)", options(nostack));  // Uppercase V register pair
        asm!("v1:0.w = vsub(v1:0.w,v1:0.w):sat", options(nostack));  // Lowercase v with suffix

        // Mixed with actual labels should still trigger for the labels
        asm!("label1: r7:6 = combine(#2, #3)"); //~ ERROR avoid using named labels

        // Regular labels should still trigger
        asm!("hexagon_label: nop"); //~ ERROR avoid using named labels
    }
}
