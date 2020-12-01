use crate::abi::call::{ArgAbi, ArgExtension, CastTarget, FnAbi, PassMode, Reg, RegKind, Uniform};
use crate::abi::{self, HasDataLayout, LayoutOf, Size, TyAndLayout, TyAndLayoutMethods};

fn extend_integer_width_mips<Ty>(arg: &mut ArgAbi<'_, Ty>, bits: u64) {
    // Always sign extend u32 values on 64-bit mips
    if let abi::Abi::Scalar(ref scalar) = arg.layout.abi {
        if let abi::Int(i, signed) = scalar.value {
            if !signed && i.size().bits() == 32 {
                if let PassMode::Direct(ref mut attrs) = arg.mode {
                    attrs.ext(ArgExtension::Sext);
                    return;
                }
            }
        }
    }

    arg.extend_integer_width_to(bits);
}

fn float_reg<'a, Ty, C>(cx: &C, ret: &ArgAbi<'a, Ty>, i: usize) -> Option<Reg>
where
    Ty: TyAndLayoutMethods<'a, C> + Copy,
    C: LayoutOf<Ty = Ty, TyAndLayout = TyAndLayout<'a, Ty>> + HasDataLayout,
{
    match ret.layout.field(cx, i).abi {
        abi::Abi::Scalar(ref scalar) => match scalar.value {
            abi::F32 => Some(Reg::f32()),
            abi::F64 => Some(Reg::f64()),
            _ => None,
        },
        _ => None,
    }
}

fn classify_ret<'a, Ty, C>(cx: &C, ret: &mut ArgAbi<'a, Ty>)
where
    Ty: TyAndLayoutMethods<'a, C> + Copy,
    C: LayoutOf<Ty = Ty, TyAndLayout = TyAndLayout<'a, Ty>> + HasDataLayout,
{
    if !ret.layout.is_aggregate() {
        extend_integer_width_mips(ret, 64);
        return;
    }

    let size = ret.layout.size;
    let bits = size.bits();
    if bits <= 128 {
        // Unlike other architectures which return aggregates in registers, MIPS n64 limits the
        // use of float registers to structures (not unions) containing exactly one or two
        // float fields.

        if let abi::FieldsShape::Arbitrary { .. } = ret.layout.fields {
            if ret.layout.fields.count() == 1 {
                if let Some(reg) = float_reg(cx, ret, 0) {
                    ret.cast_to(reg);
                    return;
                }
            } else if ret.layout.fields.count() == 2 {
                if let Some(reg0) = float_reg(cx, ret, 0) {
                    if let Some(reg1) = float_reg(cx, ret, 1) {
                        ret.cast_to(CastTarget::pair(reg0, reg1));
                        return;
                    }
                }
            }
        }

        // Cast to a uniform int structure
        ret.cast_to(Uniform { unit: Reg::i64(), total: size });
    } else {
        ret.make_indirect();
    }
}

fn classify_arg<'a, Ty, C>(cx: &C, arg: &mut ArgAbi<'a, Ty>)
where
    Ty: TyAndLayoutMethods<'a, C> + Copy,
    C: LayoutOf<Ty = Ty, TyAndLayout = TyAndLayout<'a, Ty>> + HasDataLayout,
{
    if !arg.layout.is_aggregate() {
        extend_integer_width_mips(arg, 64);
        return;
    }

    let dl = cx.data_layout();
    let size = arg.layout.size;
    let mut prefix = [None; 8];
    let mut prefix_index = 0;

    match arg.layout.fields {
        abi::FieldsShape::Primitive => unreachable!(),
        abi::FieldsShape::Array { .. } => {
            // Arrays are passed indirectly
            arg.make_indirect();
            return;
        }
        abi::FieldsShape::Union(_) => {
            // Unions and are always treated as a series of 64-bit integer chunks
        }
        abi::FieldsShape::Arbitrary { .. } => {
            // Structures are split up into a series of 64-bit integer chunks, but any aligned
            // doubles not part of another aggregate are passed as floats.
            let mut last_offset = Size::ZERO;

            for i in 0..arg.layout.fields.count() {
                let field = arg.layout.field(cx, i);
                let offset = arg.layout.fields.offset(i);

                // We only care about aligned doubles
                if let abi::Abi::Scalar(ref scalar) = field.abi {
                    if let abi::F64 = scalar.value {
                        if offset.is_aligned(dl.f64_align.abi) {
                            // Insert enough integers to cover [last_offset, offset)
                            assert!(last_offset.is_aligned(dl.f64_align.abi));
                            for _ in 0..((offset - last_offset).bits() / 64)
                                .min((prefix.len() - prefix_index) as u64)
                            {
                                prefix[prefix_index] = Some(RegKind::Integer);
                                prefix_index += 1;
                            }

                            if prefix_index == prefix.len() {
                                break;
                            }

                            prefix[prefix_index] = Some(RegKind::Float);
                            prefix_index += 1;
                            last_offset = offset + Reg::f64().size;
                        }
                    }
                }
            }
        }
    };

    // Extract first 8 chunks as the prefix
    let rest_size = size - Size::from_bytes(8) * prefix_index as u64;
    arg.cast_to(CastTarget {
        prefix,
        prefix_chunk_size: Size::from_bytes(8),
        rest: Uniform { unit: Reg::i64(), total: rest_size },
    });
}

pub fn compute_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>)
where
    Ty: TyAndLayoutMethods<'a, C> + Copy,
    C: LayoutOf<Ty = Ty, TyAndLayout = TyAndLayout<'a, Ty>> + HasDataLayout,
{
    if !fn_abi.ret.is_ignore() {
        classify_ret(cx, &mut fn_abi.ret);
    }

    for arg in &mut fn_abi.args {
        if arg.is_ignore() {
            continue;
        }
        classify_arg(cx, arg);
    }
}
