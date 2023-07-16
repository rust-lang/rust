use crate::abi::call::{ArgAbi, FnAbi, Reg, RegKind, Uniform};
use crate::abi::{HasDataLayout, TyAbiInterface};

/// Indicates the variant of the AArch64 ABI we are compiling for.
/// Used to accommodate Apple and Microsoft's deviations from the usual AAPCS ABI.
///
/// Corresponds to Clang's `AArch64ABIInfo::ABIKind`.
#[derive(Copy, Clone, PartialEq)]
pub enum AbiKind {
    AAPCS,
    DarwinPCS,
    Win64,
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
            RegKind::Vector => size.bits() == 64 || size.bits() == 128,
        };

        valid_unit.then_some(Uniform { unit, total: size })
    })
}

fn classify_ret<'a, Ty, C>(cx: &C, ret: &mut ArgAbi<'a, Ty>, kind: AbiKind)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if !ret.layout.is_aggregate() {
        if kind == AbiKind::DarwinPCS {
            // On Darwin, when returning an i8/i16, it must be sign-extended to 32 bits,
            // and likewise a u8/u16 must be zero-extended to 32-bits.
            // See also: <https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms#Pass-Arguments-to-Functions-Correctly>
            ret.extend_integer_width_to(32)
        }
        return;
    }
    if let Some(uniform) = is_homogeneous_aggregate(cx, ret) {
        ret.cast_to(uniform);
        return;
    }
    let size = ret.layout.size;
    let bits = size.bits();
    if bits <= 128 {
        ret.cast_to(Uniform { unit: Reg::i64(), total: size });
        return;
    }
    ret.make_indirect();
}

fn classify_arg<'a, Ty, C>(cx: &C, arg: &mut ArgAbi<'a, Ty>, kind: AbiKind)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if !arg.layout.is_aggregate() {
        if kind == AbiKind::DarwinPCS {
            // On Darwin, when passing an i8/i16, it must be sign-extended to 32 bits,
            // and likewise a u8/u16 must be zero-extended to 32-bits.
            // See also: <https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms#Pass-Arguments-to-Functions-Correctly>
            arg.extend_integer_width_to(32);
        }
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
            arg.cast_to(Uniform { unit: Reg::i128(), total: size });
        } else {
            arg.cast_to(Uniform { unit: Reg::i64(), total: size });
        }
        return;
    }
    arg.make_indirect();
}

pub fn compute_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>, kind: AbiKind)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
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
