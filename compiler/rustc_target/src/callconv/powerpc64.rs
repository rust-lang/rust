// FIXME:
// Alignment of 128 bit types is not currently handled, this will
// need to be fixed when PowerPC vector support is added.

use rustc_abi::{Endian, HasDataLayout, TyAbiInterface};

use crate::callconv::{Align, ArgAbi, FnAbi, Reg, RegKind, Uniform};
use crate::spec::HasTargetSpec;

#[derive(Debug, Clone, Copy, PartialEq)]
enum ABI {
    ELFv1, // original ABI used for powerpc64 (big-endian)
    ELFv2, // newer ABI used for powerpc64le and musl (both endians)
    AIX,   // used by AIX OS, big-endian only
}
use ABI::*;

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
        if ((abi == ELFv1 || abi == AIX) && arg.layout.size > unit.size)
            || arg.layout.size > unit.size.checked_mul(8, cx).unwrap()
        {
            return None;
        }

        let valid_unit = match unit.kind {
            RegKind::Integer => false,
            RegKind::Float => true,
            RegKind::Vector => arg.layout.size.bits() == 128,
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
    if !arg.layout.is_aggregate() {
        arg.extend_integer_width_to(64);
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
    let abi = if cx.target_spec().env == "musl" || cx.target_spec().os == "freebsd" {
        ELFv2
    } else if cx.target_spec().os == "aix" {
        AIX
    } else {
        match cx.data_layout().endian {
            Endian::Big => ELFv1,
            Endian::Little => ELFv2,
        }
    };

    classify(cx, &mut fn_abi.ret, abi, true);

    for arg in fn_abi.args.iter_mut() {
        classify(cx, arg, abi, false);
    }
}
