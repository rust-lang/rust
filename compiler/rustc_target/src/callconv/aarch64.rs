use std::iter;

use rustc_abi::{BackendRepr, HasDataLayout, Primitive, TyAbiInterface};

use crate::callconv::{ArgAbi, FnAbi, Reg, RegKind, Uniform};
use crate::spec::{HasTargetSpec, Target};

/// Indicates the variant of the AArch64 ABI we are compiling for.
/// Used to accommodate Apple and Microsoft's deviations from the usual AAPCS ABI.
///
/// Corresponds to Clang's `AArch64ABIInfo::ABIKind`.
#[derive(Copy, Clone, PartialEq)]
pub(crate) enum AbiKind {
    AAPCS,
    DarwinPCS,
    Win64,
}

fn is_homogeneous_aggregate<'a, Ty, C>(cx: &C, arg: &mut ArgAbi<'a, Ty>) -> Option<Uniform>
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout + HasTargetSpec,
{
    arg.layout.homogeneous_aggregate(cx).ok().and_then(|ha| ha.unit()).and_then(|unit| {
        let size = arg.layout.size;

        // Ensure we have at most four uniquely addressable members.
        if size > unit.size.checked_mul(4, cx).unwrap() {
            return None;
        }

        let valid_unit = match unit.kind {
            RegKind::Integer => false,
            // The softfloat ABI treats floats like integers, so they
            // do not get homogeneous aggregate treatment.
            RegKind::Float => cx.target_spec().abi != "softfloat",
            RegKind::Vector => size.bits() == 64 || size.bits() == 128,
        };

        valid_unit.then_some(Uniform::consecutive(unit, size))
    })
}

fn softfloat_float_abi<Ty>(target: &Target, arg: &mut ArgAbi<'_, Ty>) {
    if target.abi != "softfloat" {
        return;
    }
    // Do *not* use the float registers for passing arguments, as that would make LLVM pick the ABI
    // and its choice depends on whether `neon` instructions are enabled. Instead, we follow the
    // AAPCS "softfloat" ABI, which specifies that floats should be passed as equivalently-sized
    // integers. Nominally this only exists for "R" profile chips, but sometimes people don't want
    // to use hardfloats even if the hardware supports them, so we do this for all softfloat
    // targets.
    if let BackendRepr::Scalar(s) = arg.layout.backend_repr
        && let Primitive::Float(f) = s.primitive()
    {
        arg.cast_to(Reg { kind: RegKind::Integer, size: f.size() });
    } else if let BackendRepr::ScalarPair(s1, s2) = arg.layout.backend_repr
        && (matches!(s1.primitive(), Primitive::Float(_))
            || matches!(s2.primitive(), Primitive::Float(_)))
    {
        // This case can only be reached for the Rust ABI, so we can do whatever we want here as
        // long as it does not depend on target features (i.e., as long as we do not use float
        // registers). So we pass small things in integer registers and large things via pointer
        // indirection. This means we lose the nice "pass it as two arguments" optimization, but we
        // currently just have to way to combine a `PassMode::Cast` with that optimization (and we
        // need a cast since we want to pass the float as an int).
        if arg.layout.size.bits() <= target.pointer_width.into() {
            arg.cast_to(Reg { kind: RegKind::Integer, size: arg.layout.size });
        } else {
            arg.make_indirect();
        }
    }
}

fn classify_ret<'a, Ty, C>(cx: &C, ret: &mut ArgAbi<'a, Ty>, kind: AbiKind)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout + HasTargetSpec,
{
    if !ret.layout.is_sized() {
        // Not touching this...
        return;
    }
    if !ret.layout.is_aggregate() {
        if kind == AbiKind::DarwinPCS {
            // On Darwin, when returning an i8/i16, it must be sign-extended to 32 bits,
            // and likewise a u8/u16 must be zero-extended to 32-bits.
            // See also: <https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms#Pass-Arguments-to-Functions-Correctly>
            ret.extend_integer_width_to(32)
        }
        softfloat_float_abi(cx.target_spec(), ret);
        return;
    }
    if let Some(uniform) = is_homogeneous_aggregate(cx, ret) {
        ret.cast_to(uniform);
        return;
    }
    let size = ret.layout.size;
    let bits = size.bits();
    if bits <= 128 {
        ret.cast_to(Uniform::new(Reg::i64(), size));
        return;
    }
    ret.make_indirect();
}

fn classify_arg<'a, Ty, C>(cx: &C, arg: &mut ArgAbi<'a, Ty>, kind: AbiKind)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout + HasTargetSpec,
{
    if !arg.layout.is_sized() {
        // Not touching this...
        return;
    }
    if !arg.layout.is_aggregate() {
        if kind == AbiKind::DarwinPCS {
            // On Darwin, when passing an i8/i16, it must be sign-extended to 32 bits,
            // and likewise a u8/u16 must be zero-extended to 32-bits.
            // See also: <https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms#Pass-Arguments-to-Functions-Correctly>
            arg.extend_integer_width_to(32);
        }
        softfloat_float_abi(cx.target_spec(), arg);

        return;
    }
    if let Some(uniform) = is_homogeneous_aggregate(cx, arg) {
        arg.cast_to(uniform);
        return;
    }
    let size = arg.layout.size;
    let align = if kind == AbiKind::AAPCS {
        // When passing small aggregates by value, the AAPCS ABI mandates using the unadjusted
        // alignment of the type (not including `repr(align)`).
        // This matches behavior of `AArch64ABIInfo::classifyArgumentType` in Clang.
        // See: <https://github.com/llvm/llvm-project/blob/5e691a1c9b0ad22689d4a434ddf4fed940e58dec/clang/lib/CodeGen/TargetInfo.cpp#L5816-L5823>
        arg.layout.unadjusted_abi_align
    } else {
        arg.layout.align.abi
    };
    if size.bits() <= 128 {
        if align.bits() == 128 {
            arg.cast_to(Uniform::new(Reg::i128(), size));
        } else {
            arg.cast_to(Uniform::new(Reg::i64(), size));
        }
        return;
    }
    arg.make_indirect();
}

pub(crate) fn compute_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>, kind: AbiKind)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout + HasTargetSpec,
{
    if !fn_abi.ret.is_ignore() {
        classify_ret(cx, &mut fn_abi.ret, kind);
    }

    for arg in fn_abi.args.iter_mut() {
        if arg.is_ignore() {
            continue;
        }
        classify_arg(cx, arg, kind);
    }
}

pub(crate) fn compute_rust_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout + HasTargetSpec,
{
    for arg in fn_abi.args.iter_mut().chain(iter::once(&mut fn_abi.ret)) {
        softfloat_float_abi(cx.target_spec(), arg);
    }
}
