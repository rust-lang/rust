use rustc_abi::{
    AddressSpace, Align, BackendRepr, Float, HasDataLayout, Primitive, Reg, RegKind, TyAndLayout,
};

use crate::callconv::{
    ArgAbi, ArgAttribute, ArgAttributes, CastTarget, FnAbi, PassMode, TyAbiInterface,
};
use crate::spec::{HasTargetSpec, RustcAbi};

/// Is this a struct with a single float field?
fn is_single_fp_element<'a, Ty, C>(mut layout: TyAndLayout<'a, Ty>, cx: &C) -> bool
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    // On X86 over-aligned structs are disqualified.
    let outer_size = layout.layout.size();

    loop {
        // We're only looking for scalar types that are non-ZST.
        layout = layout.peel_transparent_wrappers_from_non_1zst(cx);

        return match layout.backend_repr {
            BackendRepr::Scalar(scalar) => match scalar.primitive() {
                Primitive::Float(float) => float.size() == outer_size,
                Primitive::Int(_, _) | Primitive::Pointer(_) => false,
            },
            BackendRepr::Memory { .. } => {
                // Structs, unions and arrays all qualify.
                if let Some((_idx, field)) = layout.non_zst_field_ignore_alignment(cx) {
                    // NOTE: alignment is not relevant here, checking for 1-ZST is incorrect.
                    layout = field;
                    continue;
                } else {
                    false
                }
            }
            _ => false,
        };
    }
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Flavor {
    General { regparam: Option<u32> },
    Fastcall,
    Vectorcall,
}

pub(crate) fn pass_on_x87_floating_point_stack<'a, C, Ty>(cx: &C, arg_abi: &mut ArgAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    let mut cast: CastTarget = match arg_abi.layout.size.bytes() {
        4 => Reg::f32().into(),
        8 => Reg::f64().into(),
        _ => unreachable!("arg must be the size of a `f32` or `f64`"),
    };
    cast.x87_floating_point_stack = true;
    // Forward whether the argument is `NoUndef` or not to improve codegen.
    cast.attrs = if let PassMode::Direct(attrs) = arg_abi.mode {
        attrs
    } else if super::layout_is_noundef(arg_abi.layout, cx) {
        ArgAttribute::NoUndef.into()
    } else {
        ArgAttributes::new()
    };
    arg_abi.mode = PassMode::Cast { pad_i32_count: 0, cast: Box::new(cast) };
}

#[derive(Clone, Copy)]
pub(crate) struct X86Options {
    pub flavor: Flavor,
    pub reg_struct_return: bool,
}

fn classify_ret<'a, Ty, C>(cx: &C, opts: X86Options, ret: &mut ArgAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout + HasTargetSpec,
{
    // "vectorcall" returns floats in `xmm0`, and soft float also does not use the x87 stack.
    let uses_x87_return = cx.target_spec().rustc_abi != Some(RustcAbi::Softfloat)
        && opts.flavor != Flavor::Vectorcall;
    if ret.layout.is_aggregate() && ret.layout.is_sized() {
        // Returning a structure. Most often, this will use
        // a hidden first argument. On some platforms, though,
        // small structs are returned as integers.
        //
        // Some links:
        // https://www.angelcode.com/dev/callconv/callconv.html
        // Clang's ABI handling is in lib/CodeGen/TargetInfo.cpp
        let t = cx.target_spec();
        if let Some(Float::F16) = ret.layout.complex_float(cx) {
            // `_Complex _Float16` is returned as `<2 x half>`.
            let kind = RegKind::Vector { hint_vector_elem: Primitive::Float(Float::F16) };
            ret.cast_to(Reg { kind, size: ret.layout.size });
        } else if t.abi_return_struct_as_int
            || opts.reg_struct_return
            || ret.layout.is_complex_number(cx)
        {
            // According to Clang, everyone but MSVC returns single-element
            // float aggregates directly in a floating-point register.
            if is_single_fp_element(ret.layout, cx) {
                match ret.layout.size.bytes() {
                    2 => ret.cast_to(Reg::f16()),
                    // The calling convention passes `f32`/`f64` returns via the x87 stack. Tell
                    // the backend to convert to an `x86_fp80` manually to avoid LLVM quieting
                    // signalling NaNs when loading/storing to/from the x87 stack.
                    4 | 8 if uses_x87_return => pass_on_x87_floating_point_stack(cx, ret),
                    4 => ret.cast_to(Reg::f32()),
                    8 => ret.cast_to(Reg::f64()),
                    _ => ret.make_indirect(),
                }
            } else {
                match ret.layout.size.bytes() {
                    1 => ret.cast_to(Reg::i8()),
                    2 => ret.cast_to(Reg::i16()),
                    4 => ret.cast_to(Reg::i32()),
                    8 => ret.cast_to(Reg::i64()),
                    _ => ret.make_indirect(),
                }
            }
        } else {
            ret.make_indirect();
        }
    } else if uses_x87_return
        && let BackendRepr::Scalar(scalar) = ret.layout.backend_repr
        && matches!(scalar.primitive(), Primitive::Float(Float::F32 | Float::F64))
    {
        pass_on_x87_floating_point_stack(cx, ret);
    } else {
        ret.extend_integer_width_to(32);
    }
}

fn classify_arg<'a, Ty, C>(cx: &C, arg: &mut ArgAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout + HasTargetSpec,
{
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
                BackendRepr::Scalar(_) | BackendRepr::ScalarPair { .. } => false,
                BackendRepr::SimdVector { .. } => true,
                BackendRepr::Memory { .. } => {
                    for i in 0..layout.fields.count() {
                        if contains_vector(cx, layout.field(cx, i)) {
                            return true;
                        }
                    }
                    false
                }
                BackendRepr::SimdScalableVector { .. } => {
                    panic!("scalable vectors are unsupported")
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

pub(crate) fn compute_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>, opts: X86Options)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout + HasTargetSpec,
{
    if !fn_abi.ret.is_ignore() {
        classify_ret(cx, opts, &mut fn_abi.ret);
    }

    for arg in fn_abi.args.iter_mut() {
        if arg.is_ignore() || !arg.layout.is_sized() {
            continue;
        }

        if arg.layout.pass_indirectly_in_non_rustic_abis(cx) {
            arg.make_indirect();
            continue;
        }

        classify_arg(cx, arg);
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
    // Mark arguments as InReg like clang does it,
    // so our fastcall/vectorcall is compatible with C/C++ fastcall/vectorcall.

    // Clang reference: lib/CodeGen/TargetInfo.cpp
    // See X86_32ABIInfo::shouldPrimitiveUseInReg(), X86_32ABIInfo::updateFreeRegs()

    // IsSoftFloatABI is only set to true on ARM platforms,
    // which in turn can't be x86?

    // The number of registers available for argument passing.
    //
    // An `extern "fastcall"` and `extern "vectorcall"` function always have 2 registers available.
    // Otherwise the `regparam` count (in the range 0..=3) determines the number of available
    // registers. If unspecified, no registers are used for argument passing.
    //
    // Functions that take a variable number of arguments continue to be passed all of their
    // arguments on the stack.
    let mut free_regs = match opts.flavor {
        _ if fn_abi.c_variadic => 0,
        Flavor::Fastcall | Flavor::Vectorcall => 2,
        Flavor::General { regparam } => u64::from(regparam.unwrap_or(0)),
    };

    if free_regs == 0 {
        return;
    }

    // For types generating PassMode::Cast, InRegs will not be set.
    // Maybe, this is a FIXME
    let has_casts = fn_abi.args.iter().any(|arg| matches!(arg.mode, PassMode::Cast { .. }));
    if has_casts && rust_abi {
        return;
    }

    for arg in fn_abi.args.iter_mut() {
        let attrs = match arg.mode {
            PassMode::Ignore | PassMode::Indirect { attrs: _, address_space: _, mode: _ } => {
                continue;
            }
            PassMode::Direct(ref mut attrs) => attrs,
            PassMode::Pair(..)
            | PassMode::IndirectUnsized { attrs: _, meta_attrs: _ }
            | PassMode::Cast { .. } => {
                unreachable!("x86 shouldn't be passing arguments by {:?}", arg.mode)
            }
        };

        // At this point we know this must be a primitive of sorts.
        let unit = arg.layout.homogeneous_aggregate(cx).unwrap().unit().unwrap();
        assert_eq!(unit.size, arg.layout.size);
        if matches!(unit.kind, RegKind::Float | RegKind::Vector { .. }) {
            continue;
        }

        let size_in_regs = arg.layout.size.bits().div_ceil(32);

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
            BackendRepr::ScalarPair { a: s1, b: s2, b_offset: _ } => {
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
                fn_abi.ret.cast_to(Reg::opaque_vector(fn_abi.ret.layout.size));
            } else if fn_abi.ret.layout.size <= Primitive::Pointer(AddressSpace::ZERO).size(cx) {
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
