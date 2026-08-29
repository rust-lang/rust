// Reference: ELF Application Binary Interface s390x Supplement
// https://github.com/IBM/s390x-abi

use rustc_abi::{BackendRepr, FieldsShape, HasDataLayout, Primitive, TyAbiInterface, TyAndLayout};

use crate::callconv::{ArgAbi, FnAbi, Reg};
use crate::spec::{Env, HasTargetSpec, Os};

/// Is this a struct with a single float field?
fn is_single_fp_element<'a, Ty, C>(mut layout: TyAndLayout<'a, Ty>, cx: &C) -> bool
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    // Contrary to X86, trailing padding is allowed on s390x.

    layout = layout.peel_transparent_wrappers(cx);
    match layout.backend_repr {
        BackendRepr::Scalar(scalar) => match scalar.primitive() {
            Primitive::Float(_) => true,
            Primitive::Int(_, _) | Primitive::Pointer(_) => false,
        },
        BackendRepr::Memory { .. } => {
            // A single-element array or union does not qualify.
            if let FieldsShape::Arbitrary { .. } = layout.fields
                && layout.fields.count() == 1
                && layout.fields.offset(0).bytes() == 0
            {
                is_single_fp_element(layout.field(cx, 0), cx)
            } else {
                false
            }
        }
        _ => false,
    }
}

fn classify_ret<Ty>(ret: &mut ArgAbi<'_, Ty>) {
    let size = ret.layout.size;
    if size.bits() <= 128 && matches!(ret.layout.backend_repr, BackendRepr::SimdVector { .. }) {
        return;
    }
    if !ret.layout.is_aggregate() && size.bits() <= 64 {
        ret.extend_integer_width_to(64);
    } else {
        ret.make_indirect();
    }
}

fn classify_arg<'a, Ty, C>(cx: &C, arg: &mut ArgAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout + HasTargetSpec,
{
    if !arg.layout.is_sized() {
        // Not touching this...
        return;
    }
    if arg.is_ignore() {
        // s390x-unknown-linux-{gnu,musl,uclibc} doesn't ignore ZSTs.
        if cx.target_spec().os == Os::Linux
            && matches!(cx.target_spec().env, Env::Gnu | Env::Musl | Env::Uclibc)
            && arg.layout.is_zst()
        {
            arg.make_indirect_from_ignore();
        }
        return;
    }
    if arg.layout.pass_indirectly_in_non_rustic_abis(cx) {
        arg.make_indirect();
        return;
    }

    if arg.layout.is_complex_number(cx) {
        arg.make_indirect();
        return;
    }

    let size = arg.layout.size;
    if size.bits() <= 128 {
        if let BackendRepr::SimdVector { .. } = arg.layout.backend_repr {
            // pass non-wrapped vector types using `PassMode::Direct`
            return;
        }

        if arg.layout.is_single_vector_element(cx, size) {
            // pass non-transparent wrappers around a vector as `PassMode::Cast`
            arg.cast_to(Reg::opaque_vector(size));
            return;
        }
    }
    if !arg.layout.is_aggregate() && size.bits() <= 64 {
        arg.extend_integer_width_to(64);
        return;
    }

    if is_single_fp_element(arg.layout, cx) {
        // Match GCC and Clang by explicitly passing padding, even though their behavior violates
        // (our reading of) the specification, which says that:
        //
        // > Structures equivalent to a floating point type are passed in floating point registers.
        // > A structure is equivalent to a floating point type if and only if it has exactly one
        // > member, which is either of floating point type of itself a structure equivalent to a
        // > floating point type.
        //
        // When the alignment is at most 8 but still overaligns the element, our implementation
        // (matching GCC and Clang) is compliant but does require suboptimally large loads and
        // stores.
        //
        // When the alignment is higher than 8, we passed the argument indirectly, which violates
        // the specification but is consistent with GCC and Clang.
        match size.bytes() {
            2 => arg.cast_to(Reg::f16()),
            4 => arg.cast_to(Reg::f32()),
            8 => arg.cast_to(Reg::f64()),
            _ => arg.make_indirect(),
        }
    } else {
        match size.bytes() {
            1 => arg.cast_to(Reg::i8()),
            2 => arg.cast_to(Reg::i16()),
            4 => arg.cast_to(Reg::i32()),
            8 => arg.cast_to(Reg::i64()),
            _ => arg.make_indirect(),
        }
    }
}

pub(crate) fn compute_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout + HasTargetSpec,
{
    if !fn_abi.ret.is_ignore() {
        classify_ret(&mut fn_abi.ret);
    }

    for arg in fn_abi.args.iter_mut() {
        classify_arg(cx, arg);
    }
}
