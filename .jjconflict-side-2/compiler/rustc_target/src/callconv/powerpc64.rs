// FIXME:
// Alignment of 128 bit types is not currently handled, this will
// need to be fixed when PowerPC vector support is added.

use rustc_abi::{FieldsShape, HasDataLayout, Integer, Numeric, TyAbiInterface, TyAndLayout};

use crate::callconv::{Align, ArgAbi, CastTarget, FnAbi, Reg, RegKind, Uniform};
use crate::spec::{HasTargetSpec, LlvmAbi, Os};

#[derive(Debug, Clone, Copy, PartialEq)]
enum ABI {
    ELFv1, // original ABI used for powerpc64 (big-endian)
    ELFv2, // newer ABI used for powerpc64le and musl (both endians)
    AIX,   // used by AIX OS, big-endian only
}
use ABI::*;

/// Whether `layout` is or contains a union.
///
/// `homogeneous_aggregate` merges the fields of a union, so it cannot tell a union from a
/// structure. This walks the layout a second time; keep the array handling here in sync with
/// `homogeneous_aggregate`. ZSTs are ignored entirely, as both GCC and Clang do.
///
/// This does not look at enums that are represented as unions at the ABI level (e.g. a
/// `#[repr(C)]` enum with fields). That does not matter here: enums can only have integer
/// discriminants, and `is_homogeneous_aggregate` rejects anything containing an integer, so an
/// enum can never be a float or vector homogeneous aggregate.
fn is_or_contains_union<'a, Ty, C>(cx: &C, layout: TyAndLayout<'a, Ty>) -> bool
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if layout.is_zst() {
        return false;
    }
    match layout.fields {
        FieldsShape::Primitive => false,
        // A `repr(transparent)` union is guaranteed to be ABI-compatible with its single
        // non-1-ZST field, so look through it instead of rejecting it.
        FieldsShape::Union(_) => {
            match layout.non_1zst_field(cx).filter(|_| layout.is_transparent()) {
                Some((_, field)) => is_or_contains_union(cx, field),
                None => true,
            }
        }
        FieldsShape::Array { .. } => is_or_contains_union(cx, layout.field(cx, 0)),
        FieldsShape::Arbitrary { .. } => {
            (0..layout.fields.count()).any(|i| is_or_contains_union(cx, layout.field(cx, i)))
        }
    }
}

fn is_homogeneous_aggregate<'a, Ty, C>(
    cx: &C,
    arg: &mut ArgAbi<'a, Ty>,
    abi: ABI,
) -> Option<Uniform>
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    arg.layout.homogeneous_aggregate(cx).ok().and_then(|ha| ha.unit()).and_then(|unit| {
        // ELFv1 and AIX only passes one-member aggregates transparently.
        // ELFv2 passes up to eight uniquely addressable members.
        if ((abi == ELFv1 || abi == AIX)
            && (arg.layout.size > unit.size || is_or_contains_union(cx, arg.layout)))
            || arg.layout.size > unit.size.checked_mul(8, cx).unwrap()
        {
            return None;
        }

        let valid_unit = match unit.kind {
            RegKind::Integer => false,
            RegKind::Float => true,
            RegKind::Vector { .. } => unit.size.bits() == 128,
        };

        valid_unit.then_some(Uniform::consecutive(unit, arg.layout.size))
    })
}

fn classify<'a, Ty, C>(cx: &C, arg: &mut ArgAbi<'a, Ty>, abi: ABI, is_ret: bool)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if arg.is_ignore() || !arg.layout.is_sized() {
        // Not touching this...
        return;
    }
    if !is_ret && arg.layout.pass_indirectly_in_non_rustic_abis(cx) {
        arg.make_indirect();
        return;
    }
    if !arg.layout.is_aggregate() {
        arg.extend_integer_width_to(64);
        return;
    }
    if let Some(component) = arg.layout.complex_number(cx) {
        if let Numeric::Int(Integer::I16, _) = component {
            // FIXME: use `PassMode::Cast` here. In LLVM 23 doing so would hit
            // https://github.com/llvm/llvm-project/issues/218676.
            return;
        }

        let reg = Reg { kind: component.reg_kind(), size: component.size() };
        arg.cast_to(CastTarget::pair(reg, reg));
        return;
    }

    // The AIX ABI expect byval for aggregates
    // See https://github.com/llvm/llvm-project/blob/main/clang/lib/CodeGen/Targets/PPC.cpp.
    // The incoming parameter is represented as a pointer in the IR,
    // the alignment is associated with the size of the register. (align 8 for 64bit)
    if !is_ret && abi == AIX {
        arg.pass_by_stack_offset(Some(Align::from_bytes(8).unwrap()));
        return;
    }

    // The ELFv1 ABI doesn't return aggregates in registers
    if is_ret && (abi == ELFv1 || abi == AIX) {
        arg.make_indirect();
        return;
    }

    if let Some(uniform) = is_homogeneous_aggregate(cx, arg, abi) {
        arg.cast_to(uniform);
        return;
    }

    let size = arg.layout.size;
    if is_ret && size.bits() > 128 {
        // Non-homogeneous aggregates larger than two doublewords are returned indirectly.
        arg.make_indirect();
    } else if size.bits() <= 64 {
        // Aggregates smaller than a doubleword should appear in
        // the least-significant bits of the parameter doubleword.
        arg.cast_to(Reg { kind: RegKind::Integer, size })
    } else {
        // Aggregates larger than i64 should be padded at the tail to fill out a whole number
        // of i64s or i128s, depending on the aggregate alignment. Always use an array for
        // this, even if there is only a single element.
        let reg = if arg.layout.align.bytes() > 8 { Reg::i128() } else { Reg::i64() };
        arg.cast_to(Uniform::consecutive(
            reg,
            size.align_to(Align::from_bytes(reg.size.bytes()).unwrap()),
        ))
    };
}

pub(crate) fn compute_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout + HasTargetSpec,
{
    let abi = match cx.target_spec().options.llvm_abiname {
        LlvmAbi::ElfV1 => ELFv1,
        LlvmAbi::ElfV2 => ELFv2,
        LlvmAbi::Unspecified if cx.target_spec().os == Os::Aix => AIX,
        // Target::check_consistency enforces that every target except AIX
        // sets llvm_abiname to either ElfV1 or ElfV2
        _ => unreachable!(),
    };

    classify(cx, &mut fn_abi.ret, abi, true);

    for arg in fn_abi.args.iter_mut() {
        classify(cx, arg, abi, false);
    }
}
