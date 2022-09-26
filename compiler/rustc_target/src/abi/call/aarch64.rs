use crate::abi::call::{ArgAbi, FnAbi, Reg, RegKind, Uniform};
use crate::abi::{HasDataLayout, TyAbiInterface};

/// Given integer-types M and register width N (e.g. M=u16 and N=32 bits), the
/// `ParamExtension` policy specifies how a uM value should be treated when
/// passed via register or stack-slot of width N. See also rust-lang/rust#97463.
#[derive(Copy, Clone, PartialEq)]
pub enum ParamExtension {
    /// Indicates that when passing an i8/i16, either as a function argument or
    /// as a return value, it must be sign-extended to 32 bits, and likewise a
    /// u8/u16 must be zero-extended to 32-bits. (This variant is here to
    /// accommodate Apple's deviation from the usual AArch64 ABI as defined by
    /// ARM.)
    ///
    /// See also: <https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms#Pass-Arguments-to-Functions-Correctly>
    ExtendTo32Bits,

    /// Indicates that no sign- nor zero-extension is performed: if a value of
    /// type with bitwidth M is passed as function argument or return value,
    /// then M bits are copied into the least significant M bits, and the
    /// remaining bits of the register (or word of memory) are untouched.
    NoExtension,
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

fn classify_ret<'a, Ty, C>(cx: &C, ret: &mut ArgAbi<'a, Ty>, param_policy: ParamExtension)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if !ret.layout.is_aggregate() {
        match param_policy {
            ParamExtension::ExtendTo32Bits => ret.extend_integer_width_to(32),
            ParamExtension::NoExtension => {}
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

fn classify_arg<'a, Ty, C>(cx: &C, arg: &mut ArgAbi<'a, Ty>, param_policy: ParamExtension)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if !arg.layout.is_aggregate() {
        match param_policy {
            ParamExtension::ExtendTo32Bits => arg.extend_integer_width_to(32),
            ParamExtension::NoExtension => {}
        }
        return;
    }
    if let Some(uniform) = is_homogeneous_aggregate(cx, arg) {
        arg.cast_to(uniform);
        return;
    }
    let size = arg.layout.size;
    let bits = size.bits();
    if bits <= 128 {
        arg.cast_to(Uniform { unit: Reg::i64(), total: size });
        return;
    }
    arg.make_indirect();
}

pub fn compute_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>, param_policy: ParamExtension)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if !fn_abi.ret.is_ignore() {
        classify_ret(cx, &mut fn_abi.ret, param_policy);
    }

    for arg in fn_abi.args.iter_mut() {
        if arg.is_ignore() {
            continue;
        }
        classify_arg(cx, arg, param_policy);
    }
}
