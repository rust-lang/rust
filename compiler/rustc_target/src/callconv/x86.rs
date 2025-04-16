use rustc_abi::{
    AddressSpace, Align, BackendRepr, HasDataLayout, Primitive, Reg, RegKind, TyAbiInterface,
    TyAndLayout,
};

use crate::callconv::{ArgAttribute, FnAbi, PassMode};
use crate::spec::{HasTargetSpec, RustcAbi};

#[derive(PartialEq)]
pub(crate) enum Flavor {
    General,
    FastcallOrVectorcall,
}

pub(crate) struct X86Options {
    pub flavor: Flavor,
    pub regparm: Option<u32>,
    pub reg_struct_return: bool,
}

pub(crate) fn compute_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>, opts: X86Options)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout + HasTargetSpec,
{
    if !fn_abi.ret.is_ignore() {
        if fn_abi.ret.layout.is_aggregate() && fn_abi.ret.layout.is_sized() {
            // Returning a structure. Most often, this will use
            // a hidden first argument. On some platforms, though,
            // small structs are returned as integers.
            //
            // Some links:
            // https://www.angelcode.com/dev/callconv/callconv.html
            // Clang's ABI handling is in lib/CodeGen/TargetInfo.cpp
            let t = cx.target_spec();
            if t.abi_return_struct_as_int || opts.reg_struct_return {
                // According to Clang, everyone but MSVC returns single-element
                // float aggregates directly in a floating-point register.
                if fn_abi.ret.layout.is_single_fp_element(cx) {
                    match fn_abi.ret.layout.size.bytes() {
                        4 => fn_abi.ret.cast_to(Reg::f32()),
                        8 => fn_abi.ret.cast_to(Reg::f64()),
                        _ => fn_abi.ret.make_indirect(),
                    }
                } else {
                    match fn_abi.ret.layout.size.bytes() {
                        1 => fn_abi.ret.cast_to(Reg::i8()),
                        2 => fn_abi.ret.cast_to(Reg::i16()),
                        4 => fn_abi.ret.cast_to(Reg::i32()),
                        8 => fn_abi.ret.cast_to(Reg::i64()),
                        _ => fn_abi.ret.make_indirect(),
                    }
                }
            } else {
                fn_abi.ret.make_indirect();
            }
        } else {
            fn_abi.ret.extend_integer_width_to(32);
        }
    }

    for arg in fn_abi.args.iter_mut() {
        if arg.is_ignore() || !arg.layout.is_sized() {
            continue;
        }

        let t = cx.target_spec();
        let align_4 = Align::from_bytes(4).unwrap();
        let align_16 = Align::from_bytes(16).unwrap();

        if arg.layout.is_aggregate() {
            // We need to compute the alignment of the `byval` argument. The rules can be found in
            // `X86_32ABIInfo::getTypeStackAlignInBytes` in Clang's `TargetInfo.cpp`. Summarized
            // here, they are:
            //
            // 1. If the natural alignment of the type is <= 4, the alignment is 4.
            //
            // 2. Otherwise, on Linux, the alignment of any vector type is the natural alignment.
            // This doesn't matter here because we only pass aggregates via `byval`, not vectors.
            //
            // 3. Otherwise, on Apple platforms, the alignment of anything that contains a vector
            // type is 16.
            //
            // 4. If none of these conditions are true, the alignment is 4.

            fn contains_vector<'a, Ty, C>(cx: &C, layout: TyAndLayout<'a, Ty>) -> bool
            where
                Ty: TyAbiInterface<'a, C> + Copy,
            {
                match layout.backend_repr {
                    BackendRepr::Scalar(_) | BackendRepr::ScalarPair(..) => false,
                    BackendRepr::SimdVector { .. } => true,
                    BackendRepr::Memory { .. } => {
                        for i in 0..layout.fields.count() {
                            if contains_vector(cx, layout.field(cx, i)) {
                                return true;
                            }
                        }
                        false
                    }
                }
            }

            let byval_align = if arg.layout.align.abi < align_4 {
                // (1.)
                align_4
            } else if t.is_like_darwin && contains_vector(cx, arg.layout) {
                // (3.)
                align_16
            } else {
                // (4.)
                align_4
            };

            arg.pass_by_stack_offset(Some(byval_align));
        } else {
            arg.extend_integer_width_to(32);
        }
    }

    fill_inregs(cx, fn_abi, opts, false);
}

pub(crate) fn fill_inregs<'a, Ty, C>(
    cx: &C,
    fn_abi: &mut FnAbi<'a, Ty>,
    opts: X86Options,
    rust_abi: bool,
) where
    Ty: TyAbiInterface<'a, C> + Copy,
{
    if opts.flavor != Flavor::FastcallOrVectorcall && opts.regparm.is_none_or(|x| x == 0) {
        return;
    }
    // Mark arguments as InReg like clang does it,
    // so our fastcall/vectorcall is compatible with C/C++ fastcall/vectorcall.

    // Clang reference: lib/CodeGen/TargetInfo.cpp
    // See X86_32ABIInfo::shouldPrimitiveUseInReg(), X86_32ABIInfo::updateFreeRegs()

    // IsSoftFloatABI is only set to true on ARM platforms,
    // which in turn can't be x86?

    // 2 for fastcall/vectorcall, regparm limited by 3 otherwise
    let mut free_regs = opts.regparm.unwrap_or(2).into();

    // For types generating PassMode::Cast, InRegs will not be set.
    // Maybe, this is a FIXME
    let has_casts = fn_abi.args.iter().any(|arg| matches!(arg.mode, PassMode::Cast { .. }));
    if has_casts && rust_abi {
        return;
    }

    for arg in fn_abi.args.iter_mut() {
        let attrs = match arg.mode {
            PassMode::Ignore | PassMode::Indirect { attrs: _, meta_attrs: None, on_stack: _ } => {
                continue;
            }
            PassMode::Direct(ref mut attrs) => attrs,
            PassMode::Pair(..)
            | PassMode::Indirect { attrs: _, meta_attrs: Some(_), on_stack: _ }
            | PassMode::Cast { .. } => {
                unreachable!("x86 shouldn't be passing arguments by {:?}", arg.mode)
            }
        };

        // At this point we know this must be a primitive of sorts.
        let unit = arg.layout.homogeneous_aggregate(cx).unwrap().unit().unwrap();
        assert_eq!(unit.size, arg.layout.size);
        if matches!(unit.kind, RegKind::Float | RegKind::Vector) {
            continue;
        }

        let size_in_regs = (arg.layout.size.bits() + 31) / 32;

        if size_in_regs == 0 {
            continue;
        }

        if size_in_regs > free_regs {
            break;
        }

        free_regs -= size_in_regs;

        if arg.layout.size.bits() <= 32 && unit.kind == RegKind::Integer {
            attrs.set(ArgAttribute::InReg);
        }

        if free_regs == 0 {
            break;
        }
    }
}

pub(crate) fn compute_rust_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout + HasTargetSpec,
{
    // Avoid returning floats in x87 registers on x86 as loading and storing from x87
    // registers will quiet signalling NaNs. Also avoid using SSE registers since they
    // are not always available (depending on target features).
    if !fn_abi.ret.is_ignore() {
        let has_float = match fn_abi.ret.layout.backend_repr {
            BackendRepr::Scalar(s) => matches!(s.primitive(), Primitive::Float(_)),
            BackendRepr::ScalarPair(s1, s2) => {
                matches!(s1.primitive(), Primitive::Float(_))
                    || matches!(s2.primitive(), Primitive::Float(_))
            }
            _ => false, // anyway not passed via registers on x86
        };
        if has_float {
            if cx.target_spec().rustc_abi == Some(RustcAbi::X86Sse2)
                && fn_abi.ret.layout.backend_repr.is_scalar()
                && fn_abi.ret.layout.size.bits() <= 128
            {
                // This is a single scalar that fits into an SSE register, and the target uses the
                // SSE ABI. We prefer this over integer registers as float scalars need to be in SSE
                // registers for float operations, so that's the best place to pass them around.
                fn_abi.ret.cast_to(Reg { kind: RegKind::Vector, size: fn_abi.ret.layout.size });
            } else if fn_abi.ret.layout.size <= Primitive::Pointer(AddressSpace::DATA).size(cx) {
                // Same size or smaller than pointer, return in an integer register.
                fn_abi.ret.cast_to(Reg { kind: RegKind::Integer, size: fn_abi.ret.layout.size });
            } else {
                // Larger than a pointer, return indirectly.
                fn_abi.ret.make_indirect();
            }
            return;
        }
    }
}
