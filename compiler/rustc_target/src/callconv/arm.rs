use rustc_abi::{ArmCall, CanonAbi, HasDataLayout, TyAbiInterface};

use crate::callconv::{ArgAbi, FnAbi, Reg, RegKind, Uniform};
use crate::spec::HasTargetSpec;

fn is_homogeneous_aggregate<'a, Ty, C>(cx: &C, arg: &mut ArgAbi<'a, Ty>) -> Option<Uniform>
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    arg.layout.homogeneous_aggregate(cx).ok().and_then(|ha| ha.unit()).and_then(|unit| {
        let size = arg.layout.size;

        // Ensure we have at most four uniquely addressable members.
        if size > unit.size.checked_mul(4, cx).unwrap() {
            return None;
        }

        let valid_unit = match unit.kind {
            RegKind::Integer => false,
            RegKind::Float => true,
            RegKind::Vector => size.bits() == 64 || size.bits() == 128,
        };

        valid_unit.then_some(Uniform::consecutive(unit, size))
    })
}

fn classify_ret<'a, Ty, C>(cx: &C, ret: &mut ArgAbi<'a, Ty>, vfp: bool)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if !ret.layout.is_sized() {
        // Not touching this...
        return;
    }
    if !ret.layout.is_aggregate() {
        ret.extend_integer_width_to(32);
        return;
    }

    if vfp {
        if let Some(uniform) = is_homogeneous_aggregate(cx, ret) {
            ret.cast_to(uniform);
            return;
        }
    }

    let size = ret.layout.size;
    let bits = size.bits();
    if bits <= 32 {
        ret.cast_to(Uniform::new(Reg::i32(), size));
        return;
    }
    ret.make_indirect();
}

fn classify_arg<'a, Ty, C>(cx: &C, arg: &mut ArgAbi<'a, Ty>, vfp: bool)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if !arg.layout.is_sized() {
        // Not touching this...
        return;
    }
    if !arg.layout.is_aggregate() {
        arg.extend_integer_width_to(32);
        return;
    }

    if vfp {
        if let Some(uniform) = is_homogeneous_aggregate(cx, arg) {
            arg.cast_to(uniform);
            return;
        }
    }

    let align = arg.layout.align.abi.bytes();
    let total = arg.layout.size;
    arg.cast_to(Uniform::consecutive(if align <= 4 { Reg::i32() } else { Reg::i64() }, total));
}

pub(crate) fn compute_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout + HasTargetSpec,
{
    // If this is a target with a hard-float ABI, and the function is not explicitly
    // `extern "aapcs"`, then we must use the VFP registers for homogeneous aggregates.
    let vfp = cx.target_spec().llvm_target.ends_with("hf")
        && fn_abi.conv != CanonAbi::Arm(ArmCall::Aapcs)
        && !fn_abi.c_variadic;

    if !fn_abi.ret.is_ignore() {
        classify_ret(cx, &mut fn_abi.ret, vfp);
    }

    for arg in fn_abi.args.iter_mut() {
        if arg.is_ignore() {
            continue;
        }
        classify_arg(cx, arg, vfp);
    }
}
