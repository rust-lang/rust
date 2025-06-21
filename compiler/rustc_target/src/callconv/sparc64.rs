use rustc_abi::{
    Align, BackendRepr, FieldsShape, Float, HasDataLayout, Primitive, Reg, Size, TyAbiInterface,
    TyAndLayout, Variants,
};

use crate::callconv::{ArgAbi, ArgAttribute, CastTarget, FnAbi, Uniform};
use crate::spec::HasTargetSpec;

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
    match arg_layout.backend_repr {
        BackendRepr::Scalar(scalar) => match scalar.primitive() {
            Primitive::Float(float)
                if offset.is_aligned(float.align(cx).abi.min(Align::from_bytes(8).unwrap())) =>
            {
                let index = offset.bytes_usize() / 8;
                match float {
                    Float::F128 => {
                        double_words[index] = DoubleWord::F128Start;
                        double_words[index + 1] = DoubleWord::F128End;
                    }
                    Float::F64 => {
                        double_words[index] = DoubleWord::F64;
                    }
                    Float::F32 => {
                        if let DoubleWord::Words(words) = &mut double_words[index] {
                            words[(offset.bytes_usize() % 8) / 4] = Word::F32;
                        } else {
                            unreachable!();
                        }
                    }
                    Float::F16 => {
                        // FIXME(llvm/llvm-project#97981): f16 doesn't have a proper ABI in LLVM on
                        // sparc64 yet. Once it does, double-check if it needs to be passed in a
                        // floating-point register here.
                    }
                }
            }
            _ => {}
        },
        BackendRepr::SimdVector { .. } => {}
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
    let pad = !total_double_word_count.is_multiple_of(2) && arg.layout.align.abi.bytes() == 16;
    let double_word_count = arg.layout.size.bytes_usize().div_ceil(8);
    let start_double_word_count = *total_double_word_count + usize::from(pad);
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

    let mut double_words = [DoubleWord::Words([Word::Integer; 2]); 4];
    classify(cx, &arg.layout, Size::ZERO, &mut double_words);

    let mut regs = [None; 8];
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

    if let [Some(reg), None, ..] = regs {
        arg.cast_to_and_pad_i32(CastTarget::from(reg).with_attrs(attrs.into()), pad);
    } else {
        arg.cast_to_and_pad_i32(
            CastTarget::prefixed(regs, Uniform::new(Reg::i8(), Size::ZERO))
                .with_attrs(attrs.into()),
            pad,
        );
    }
}

pub(crate) fn compute_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout + HasTargetSpec,
{
    if !fn_abi.ret.is_ignore() {
        classify_arg(cx, &mut fn_abi.ret, Size::from_bytes(32), &mut 0);
    }

    let mut double_word_count = 0;
    for arg in fn_abi.args.iter_mut() {
        if arg.is_ignore() {
            // sparc64-unknown-linux-{gnu,musl,uclibc} doesn't ignore ZSTs.
            if cx.target_spec().os == "linux"
                && matches!(&*cx.target_spec().env, "gnu" | "musl" | "uclibc")
                && arg.layout.is_zst()
            {
                arg.make_indirect_from_ignore();
                double_word_count += 1;
            }
            continue;
        }
        classify_arg(cx, arg, Size::from_bytes(16), &mut double_word_count);
    }
}
