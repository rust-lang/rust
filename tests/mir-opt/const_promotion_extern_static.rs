//@ skip-filecheck
//@ ignore-endian-big
static Y: i32 = 42;

// EMIT_MIR const_promotion_extern_static.BAR.PromoteTemps.diff
// EMIT_MIR const_promotion_extern_static.BAR-promoted[0].SimplifyCfg-pre-optimizations.after.mir
static mut BAR: *const &i32 = [&Y].as_ptr();

// EMIT_MIR const_promotion_extern_static.BOP.built.after.mir
static BOP: &i32 = &13;

fn main() {}
