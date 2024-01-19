// unit-test: GVN
// EMIT_MIR cast.main.GVN.diff

fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: debug x => const 42_u32;
    // CHECK: debug y => const 42_u8;
    let x = 42u8 as u32;
    let y = 42u32 as u8;
}
