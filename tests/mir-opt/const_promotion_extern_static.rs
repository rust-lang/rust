//@ ignore-endian-big
static Y: i32 = 42;

// EMIT_MIR const_promotion_extern_static.BAR.PromoteTemps.diff
// EMIT_MIR const_promotion_extern_static.BAR-promoted[0].SimplifyCfg-pre-optimizations.after.mir
// CHECK-LABEL: static mut BAR:
// CHECK: const BAR::promoted[0];
static mut BAR: *const &i32 = [&Y].as_ptr();

// EMIT_MIR const_promotion_extern_static.BOP.built.after.mir
// CHECK-LABEL: static BOP:
// CHECK: const BOP::promoted[0];
static BOP: &i32 = &13;

fn main() {}
