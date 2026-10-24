use rustc_abi::{ArmCall, CanonAbi, HasDataLayout, TyAbiInterface};

use crate::callconv::{ArgAbi, FnAbi, Reg, RegKind, Uniform};
use crate::spec::{CfgAbi, HasTargetSpec, Os};

#[derive(Clone, Copy)]
enum ArmAbiKind {
    Aapcs,
    AapcsVfp,
    Aapcs16Vfp,
}

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
            RegKind::Vector { .. } => unit.size.bits() == 64 || unit.size.bits() == 128,
        };

        valid_unit.then_some(Uniform::consecutive(unit, size))
    })
}

fn classify_ret<'a, Ty, C>(cx: &C, ret: &mut ArgAbi<'a, Ty>, abi_kind: ArmAbiKind, vfp: bool)
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
        // Aggregates <= 4 bytes are returned in r0; other aggregates are returned indirectly.
        ret.cast_to(Uniform::new(Reg::i32(), size));
        return;
    } else if matches!(abi_kind, ArmAbiKind::Aapcs16Vfp) && bits <= 128 {
        // watchOS returns the remaining aggregates of up to 128 bits in GPRs.
        ret.cast_to(Uniform::consecutive(Reg::i32(), size));
        return;
    }

    ret.make_indirect();
}

fn classify_arg<'a, Ty, C>(cx: &C, arg: &mut ArgAbi<'a, Ty>, abi_kind: ArmAbiKind, vfp: bool)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if !arg.layout.is_sized() {
        // Not touching this...
        return;
    }
    if arg.layout.pass_indirectly_in_non_rustic_abis(cx) {
        arg.make_indirect();
        return;
    }
    if !arg.layout.is_aggregate() {
        arg.extend_integer_width_to(32);
        return;
    }

    // watchOS also passes homogeneous aggregates in VFP registers, and unlike `AapcsVfp` it does
    // so even for variadics and for `extern "aapcs"`: the backend will use GPRs if needed.
    if vfp || matches!(abi_kind, ArmAbiKind::Aapcs16Vfp) {
        if let Some(uniform) = is_homogeneous_aggregate(cx, arg) {
            arg.cast_to(uniform);
            return;
        }
    }

    // For the composites that are left, watchOS adopts the 64-bit AAPCS rule: those larger than
    // 128 bits are placed in space allocated by the caller, and a pointer is passed.
    if matches!(abi_kind, ArmAbiKind::Aapcs16Vfp) && arg.layout.size.bits() > 128 {
        arg.make_indirect();
        return;
    }

    let align = match abi_kind {
        ArmAbiKind::Aapcs | ArmAbiKind::AapcsVfp => arg.layout.unadjusted_abi_align.bytes(),
        ArmAbiKind::Aapcs16Vfp => arg.layout.align.bytes(),
    };

    let total = arg.layout.size;
    arg.cast_to(Uniform::consecutive(if align <= 4 { Reg::i32() } else { Reg::i64() }, total));
}

pub(crate) fn compute_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout + HasTargetSpec,
{
    let abi_kind = if cx.target_spec().os == Os::WatchOs {
        ArmAbiKind::Aapcs16Vfp
    } else if cx.target_spec().cfg_abi == CfgAbi::EabiHf {
        ArmAbiKind::AapcsVfp
    } else {
        ArmAbiKind::Aapcs
    };

    // Whether we must use the VFP registers for homogeneous aggregates.
    let is_effectively_vfp = |accept_aapcs16| {
        // When the user requested aapcs explicitly, honor that.
        if matches!(fn_abi.conv, CanonAbi::Arm(ArmCall::Aapcs)) {
            return false;
        }

        match abi_kind {
            ArmAbiKind::AapcsVfp => true,
            ArmAbiKind::Aapcs16Vfp => accept_aapcs16,
            ArmAbiKind::Aapcs => false,
        }
    };

    if !fn_abi.ret.is_ignore() {
        classify_ret(cx, &mut fn_abi.ret, abi_kind, !fn_abi.c_variadic && is_effectively_vfp(true));
    }

    let is_arg_vfp = !fn_abi.c_variadic && is_effectively_vfp(false);
    for arg in fn_abi.args.iter_mut() {
        if arg.is_ignore() {
            continue;
        }
        classify_arg(cx, arg, abi_kind, is_arg_vfp);
    }
}
