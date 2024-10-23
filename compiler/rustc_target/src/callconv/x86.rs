use crate::abi::call::{ArgAttribute, FnAbi, PassMode, Reg, RegKind};
use crate::abi::{
    Abi, AddressSpace, Align, Float, HasDataLayout, Pointer, TyAbiInterface, TyAndLayout,
};
use crate::spec::HasTargetSpec;
use crate::spec::abi::Abi as SpecAbi;

#[derive(PartialEq)]
pub(crate) enum Flavor {
    General,
    FastcallOrVectorcall,
}

pub(crate) struct X86Options {
    pub flavor: Flavor,
    pub regparm: Option<u32>,
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
            if t.abi_return_struct_as_int {
                // According to Clang, everyone but MSVC returns single-element
                // float aggregates directly in a floating-point register.
                if !t.is_like_msvc && fn_abi.ret.layout.is_single_fp_element(cx) {
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

        // FIXME: MSVC 2015+ will pass the first 3 vector arguments in [XYZ]MM0-2
        // See https://reviews.llvm.org/D72114 for Clang behavior

        let t = cx.target_spec();
        let align_4 = Align::from_bytes(4).unwrap();
        let align_16 = Align::from_bytes(16).unwrap();

        if t.is_like_msvc
            && arg.layout.is_adt()
            && let Some(max_repr_align) = arg.layout.max_repr_align
            && max_repr_align > align_4
        {
            // MSVC has special rules for overaligned arguments: https://reviews.llvm.org/D72114.
            // Summarized here:
            // - Arguments with _requested_ alignment > 4 are passed indirectly.
            // - For backwards compatibility, arguments with natural alignment > 4 are still passed
            //   on stack (via `byval`). For example, this includes `double`, `int64_t`,
            //   and structs containing them, provided they lack an explicit alignment attribute.
            assert!(
                arg.layout.align.abi >= max_repr_align,
                "abi alignment {:?} less than requested alignment {max_repr_align:?}",
                arg.layout.align.abi,
            );
            arg.make_indirect();
        } else if arg.layout.is_aggregate() {
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
                match layout.abi {
                    Abi::Uninhabited | Abi::Scalar(_) | Abi::ScalarPair(..) => false,
                    Abi::Vector { .. } => true,
                    Abi::Aggregate { .. } => {
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
            } else if t.is_like_osx && contains_vector(cx, arg.layout) {
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

pub(crate) fn compute_rust_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>, abi: SpecAbi)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout + HasTargetSpec,
{
    // Avoid returning floats in x87 registers on x86 as loading and storing from x87
    // registers will quiet signalling NaNs. Also avoid using SSE registers since they
    // are not always available (depending on target features).
    if !fn_abi.ret.is_ignore()
        // Intrinsics themselves are not actual "real" functions, so theres no need to change their ABIs.
        && abi != SpecAbi::RustIntrinsic
    {
        let has_float = match fn_abi.ret.layout.abi {
            Abi::Scalar(s) => matches!(s.primitive(), Float(_)),
            Abi::ScalarPair(s1, s2) => {
                matches!(s1.primitive(), Float(_)) || matches!(s2.primitive(), Float(_))
            }
            _ => false, // anyway not passed via registers on x86
        };
        if has_float {
            if fn_abi.ret.layout.size <= Pointer(AddressSpace::DATA).size(cx) {
                // Same size or smaller than pointer, return in a register.
                fn_abi.ret.cast_to(Reg { kind: RegKind::Integer, size: fn_abi.ret.layout.size });
            } else {
                // Larger than a pointer, return indirectly.
                fn_abi.ret.make_indirect();
            }
            return;
        }
    }
}
