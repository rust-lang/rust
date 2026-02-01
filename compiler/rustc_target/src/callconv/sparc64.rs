use rustc_abi::{
    Align, BackendRepr, FieldsShape, Float, HasDataLayout, Primitive, Reg, Size, TyAbiInterface,
    TyAndLayout, Variants,
};

use crate::callconv::{ArgAbi, ArgAttribute, CastTarget, FnAbi, Uniform};
use crate::spec::{Env, HasTargetSpec, Os};

// NOTE: GCC and Clang/LLVM have disagreements that the ABI doesn't resolve, we match the
// Clang/LLVM behavior in these cases.

#[derive(Copy, Clone)]
enum DoubleWord {
    F64,
    F128Start,
    F128End,
    Words([Word; 2]),
}

#[derive(Copy, Clone)]
enum Word {
    F32,
    Integer,
}

fn classify<'a, Ty, C>(
    cx: &C,
    arg_layout: &TyAndLayout<'a, Ty>,
    offset: Size,
    double_words: &mut [DoubleWord; 4],
) where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    // If this function does not update the `double_words` array, the value will be passed via
    // integer registers. The array is initialized with `DoubleWord::Words([Word::Integer; 2])`.

    match arg_layout.backend_repr {
        BackendRepr::Scalar(scalar) => match scalar.primitive() {
            Primitive::Float(float) => {
                if offset.is_aligned(Ord::min(*float.align(cx), Align::EIGHT)) {
                    let index = offset.bytes_usize() / 8;
                    match float {
                        Float::F128 => {
                            double_words[index] = DoubleWord::F128Start;
                            double_words[index + 1] = DoubleWord::F128End;
                        }
                        Float::F64 => {
                            double_words[index] = DoubleWord::F64;
                        }
                        Float::F32 => match &mut double_words[index] {
                            DoubleWord::Words(words) => {
                                words[(offset.bytes_usize() % 8) / 4] = Word::F32;
                            }
                            _ => unreachable!(),
                        },
                        Float::F16 => {
                            // Match LLVM by passing `f16` in integer registers.
                        }
                    }
                } else {
                    /* pass unaligned floats in integer registers */
                }
            }
            Primitive::Int(_, _) | Primitive::Pointer(_) => { /* pass in integer registers */ }
        },
        BackendRepr::SimdVector { .. } => {}
        BackendRepr::ScalableVector { .. } => {}
        BackendRepr::ScalarPair(..) | BackendRepr::Memory { .. } => match arg_layout.fields {
            FieldsShape::Primitive => {
                unreachable!("aggregates can't have `FieldsShape::Primitive`")
            }
            FieldsShape::Union(_) => {
                if !arg_layout.is_zst() {
                    if arg_layout.is_transparent() {
                        let non_1zst_elem = arg_layout.non_1zst_field(cx).expect("not exactly one non-1-ZST field in non-ZST repr(transparent) union").1;
                        classify(cx, &non_1zst_elem, offset, double_words);
                    }
                }
            }
            FieldsShape::Array { .. } => {}
            FieldsShape::Arbitrary { .. } => match arg_layout.variants {
                Variants::Multiple { .. } => {}
                Variants::Single { .. } | Variants::Empty => {
                    // Match Clang by ignoring whether a struct is packed and just considering
                    // whether individual fields are aligned. GCC currently uses only integer
                    // registers when passing packed structs.
                    for i in arg_layout.fields.index_by_increasing_offset() {
                        classify(
                            cx,
                            &arg_layout.field(cx, i),
                            offset + arg_layout.fields.offset(i),
                            double_words,
                        );
                    }
                }
            },
        },
    }
}

fn classify_arg<'a, Ty, C>(
    cx: &C,
    arg: &mut ArgAbi<'a, Ty>,
    in_registers_max: Size,
    total_double_word_count: &mut usize,
) where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    // 64-bit SPARC allocates argument stack space in 64-bit chunks (double words), some of which
    // are promoted to registers based on their position on the stack.

    // Keep track of the total number of double words used by arguments so far. This allows padding
    // arguments to be inserted where necessary to ensure that 16-aligned arguments are passed in an
    // aligned set of registers.

    let pad = !total_double_word_count.is_multiple_of(2) && arg.layout.align.abi.bytes() == 16;
    // The number of double words used by this argument.
    let double_word_count = arg.layout.size.bytes_usize().div_ceil(8);
    // The number of double words before this argument, including any padding.
    let start_double_word_count = *total_double_word_count + usize::from(pad);

    if arg.layout.pass_indirectly_in_non_rustic_abis(cx) {
        arg.make_indirect();
        *total_double_word_count += 1;
        return;
    }

    if !arg.layout.is_aggregate() {
        arg.extend_integer_width_to(64);
        *total_double_word_count = start_double_word_count + double_word_count;
        return;
    }

    let total = arg.layout.size;
    if total > in_registers_max {
        arg.make_indirect();
        *total_double_word_count += 1;
        return;
    }

    *total_double_word_count = start_double_word_count + double_word_count;

    const ARGUMENT_REGISTERS: usize = 8;

    let mut double_words = [DoubleWord::Words([Word::Integer; 2]); ARGUMENT_REGISTERS / 2];
    classify(cx, &arg.layout, Size::ZERO, &mut double_words);

    let mut regs = [None; ARGUMENT_REGISTERS];
    let mut i = 0;
    let mut push = |reg| {
        regs[i] = Some(reg);
        i += 1;
    };
    let mut attrs = ArgAttribute::empty();

    for (index, double_word) in double_words.into_iter().enumerate() {
        if arg.layout.size.bytes_usize() <= index * 8 {
            break;
        }
        match double_word {
            // `f128` must be aligned to be assigned a float register.
            DoubleWord::F128Start if (start_double_word_count + index).is_multiple_of(2) => {
                push(Reg::f128());
            }
            DoubleWord::F128Start => {
                // Clang currently handles this case nonsensically, always returning a packed
                // `struct { long double x; }` in an aligned quad floating-point register even when
                // the `long double` isn't aligned on the stack, which also makes all future
                // arguments get passed in the wrong registers. This passes the `f128` in integer
                // registers when it is unaligned, same as with `f32` and `f64`.
                push(Reg::i64());
                push(Reg::i64());
            }
            DoubleWord::F128End => {} // Already handled by `F128Start`
            DoubleWord::F64 => push(Reg::f64()),
            DoubleWord::Words([Word::Integer, Word::Integer]) => push(Reg::i64()),
            DoubleWord::Words(words) => {
                attrs |= ArgAttribute::InReg;
                for word in words {
                    match word {
                        Word::F32 => push(Reg::f32()),
                        Word::Integer => push(Reg::i32()),
                    }
                }
            }
        }
    }

    let cast_target = match regs {
        [Some(reg), None, ..] => CastTarget::from(reg),
        _ => CastTarget::prefixed(regs, Uniform::new(Reg::i8(), Size::ZERO)),
    };

    arg.cast_to_and_pad_i32(cast_target.with_attrs(attrs.into()), pad);
}

pub(crate) fn compute_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout + HasTargetSpec,
{
    if !fn_abi.ret.is_ignore() && fn_abi.ret.layout.is_sized() {
        // A return value of 32 bytes or smaller is passed via registers.
        classify_arg(cx, &mut fn_abi.ret, Size::from_bytes(32), &mut 0);
    }

    // sparc64-unknown-linux-{gnu,musl,uclibc} doesn't ignore ZSTs.
    let passes_zsts = match cx.target_spec().os {
        Os::Linux => matches!(cx.target_spec().env, Env::Gnu | Env::Musl | Env::Uclibc),
        _ => false,
    };

    let mut double_word_count = 0;
    for arg in fn_abi.args.iter_mut() {
        if !arg.layout.is_sized() {
            continue;
        }
        if arg.is_ignore() {
            if passes_zsts && arg.layout.is_zst() {
                arg.make_indirect_from_ignore();
                double_word_count += 1;
            }
            continue;
        }
        // An argument of 16 bytes or smaller is passed via registers.
        classify_arg(cx, arg, Size::from_bytes(16), &mut double_word_count);
    }
}
