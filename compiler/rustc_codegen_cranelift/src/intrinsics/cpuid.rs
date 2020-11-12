//! Emulation of a subset of the cpuid x86 instruction.

use crate::prelude::*;

/// Emulates a subset of the cpuid x86 instruction.
///
/// This emulates an intel cpu with sse and sse2 support, but which doesn't support anything else.
pub(crate) fn codegen_cpuid_call<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Module>,
    leaf: Value,
    _subleaf: Value,
) -> (Value, Value, Value, Value) {
    let leaf_0 = fx.bcx.create_block();
    let leaf_1 = fx.bcx.create_block();
    let leaf_8000_0000 = fx.bcx.create_block();
    let leaf_8000_0001 = fx.bcx.create_block();
    let unsupported_leaf = fx.bcx.create_block();

    let dest = fx.bcx.create_block();
    let eax = fx.bcx.append_block_param(dest, types::I32);
    let ebx = fx.bcx.append_block_param(dest, types::I32);
    let ecx = fx.bcx.append_block_param(dest, types::I32);
    let edx = fx.bcx.append_block_param(dest, types::I32);

    let mut switch = cranelift_frontend::Switch::new();
    switch.set_entry(0, leaf_0);
    switch.set_entry(1, leaf_1);
    switch.set_entry(0x8000_0000, leaf_8000_0000);
    switch.set_entry(0x8000_0001, leaf_8000_0001);
    switch.emit(&mut fx.bcx, leaf, unsupported_leaf);

    fx.bcx.switch_to_block(leaf_0);
    let max_basic_leaf = fx.bcx.ins().iconst(types::I32, 1);
    let vend0 = fx
        .bcx
        .ins()
        .iconst(types::I32, i64::from(u32::from_le_bytes(*b"Genu")));
    let vend2 = fx
        .bcx
        .ins()
        .iconst(types::I32, i64::from(u32::from_le_bytes(*b"ineI")));
    let vend1 = fx
        .bcx
        .ins()
        .iconst(types::I32, i64::from(u32::from_le_bytes(*b"ntel")));
    fx.bcx
        .ins()
        .jump(dest, &[max_basic_leaf, vend0, vend1, vend2]);

    fx.bcx.switch_to_block(leaf_1);
    let cpu_signature = fx.bcx.ins().iconst(types::I32, 0);
    let additional_information = fx.bcx.ins().iconst(types::I32, 0);
    let ecx_features = fx.bcx.ins().iconst(types::I32, 0);
    let edx_features = fx
        .bcx
        .ins()
        .iconst(types::I32, 1 << 25 /* sse */ | 1 << 26 /* sse2 */);
    fx.bcx.ins().jump(
        dest,
        &[
            cpu_signature,
            additional_information,
            ecx_features,
            edx_features,
        ],
    );

    fx.bcx.switch_to_block(leaf_8000_0000);
    let extended_max_basic_leaf = fx.bcx.ins().iconst(types::I32, 0);
    let zero = fx.bcx.ins().iconst(types::I32, 0);
    fx.bcx
        .ins()
        .jump(dest, &[extended_max_basic_leaf, zero, zero, zero]);

    fx.bcx.switch_to_block(leaf_8000_0001);
    let zero = fx.bcx.ins().iconst(types::I32, 0);
    let proc_info_ecx = fx.bcx.ins().iconst(types::I32, 0);
    let proc_info_edx = fx.bcx.ins().iconst(types::I32, 0);
    fx.bcx
        .ins()
        .jump(dest, &[zero, zero, proc_info_ecx, proc_info_edx]);

    fx.bcx.switch_to_block(unsupported_leaf);
    crate::trap::trap_unreachable(
        fx,
        "__cpuid_count arch intrinsic doesn't yet support specified leaf",
    );

    fx.bcx.switch_to_block(dest);
    fx.bcx.ins().nop();

    (eax, ebx, ecx, edx)
}
