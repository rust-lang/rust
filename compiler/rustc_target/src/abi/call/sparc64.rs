// FIXME: This needs an audit for correctness and completeness.

use crate::abi::call::{
    ArgAbi, ArgAttribute, ArgAttributes, ArgExtension, CastTarget, FnAbi, Reg, RegKind, Uniform,
};
use crate::abi::{self, HasDataLayout, Size, TyAbiInterface};

fn is_homogeneous_aggregate<'a, Ty, C>(cx: &C, arg: &mut ArgAbi<'a, Ty>) -> Option<Uniform>
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    arg.layout.homogeneous_aggregate(cx).ok().and_then(|ha| ha.unit()).and_then(|unit| {
        // Ensure we have at most eight uniquely addressable members.
        if arg.layout.size > unit.size.checked_mul(8, cx).unwrap() {
            return None;
        }

        let valid_unit = match unit.kind {
            RegKind::Integer => false,
            RegKind::Float => false,
            RegKind::Vector => arg.layout.size.bits() == 128,
        };

        valid_unit.then_some(Uniform { unit, total: arg.layout.size })
    })
}

fn classify_arg<'a, Ty, C>(cx: &C, arg: &mut ArgAbi<'a, Ty>, in_registers_max: Size)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if !arg.layout.is_aggregate() {
        arg.extend_integer_width_to(64);
        return;
    }

    // This doesn't intentionally handle structures with floats which needs
    // special care below.
    if let Some(uniform) = is_homogeneous_aggregate(cx, arg) {
        arg.cast_to(uniform);
        return;
    }

    if let abi::FieldsShape::Arbitrary { .. } = arg.layout.fields {
        let dl = cx.data_layout();
        let size = arg.layout.size;
        let mut prefix = [None; 8];
        let mut prefix_index = 0;
        let mut last_offset = Size::ZERO;
        let mut has_float = false;
        let mut arg_attribute = ArgAttribute::default();

        for i in 0..arg.layout.fields.count() {
            let field = arg.layout.field(cx, i);
            let offset = arg.layout.fields.offset(i);

            if let abi::Abi::Scalar(scalar) = &field.abi {
                if scalar.value == abi::F32 || scalar.value == abi::F64 {
                    has_float = true;

                    if !last_offset.is_aligned(dl.f64_align.abi) && last_offset < offset {
                        if prefix_index == prefix.len() {
                            break;
                        }
                        prefix[prefix_index] = Some(Reg::i32());
                        prefix_index += 1;
                        last_offset = last_offset + Reg::i32().size;
                    }

                    for _ in 0..((offset - last_offset).bits() / 64)
                        .min((prefix.len() - prefix_index) as u64)
                    {
                        prefix[prefix_index] = Some(Reg::i64());
                        prefix_index += 1;
                        last_offset = last_offset + Reg::i64().size;
                    }

                    if last_offset < offset {
                        if prefix_index == prefix.len() {
                            break;
                        }
                        prefix[prefix_index] = Some(Reg::i32());
                        prefix_index += 1;
                        last_offset = last_offset + Reg::i32().size;
                    }

                    if prefix_index == prefix.len() {
                        break;
                    }

                    if scalar.value == abi::F32 {
                        arg_attribute = ArgAttribute::InReg;
                        prefix[prefix_index] = Some(Reg::f32());
                        last_offset = offset + Reg::f32().size;
                    } else {
                        prefix[prefix_index] = Some(Reg::f64());
                        last_offset = offset + Reg::f64().size;
                    }
                    prefix_index += 1;
                }
            }
        }

        if has_float && arg.layout.size <= in_registers_max {
            let mut rest_size = size - last_offset;

            if (rest_size.raw % 8) != 0 && prefix_index < prefix.len() {
                prefix[prefix_index] = Some(Reg::i32());
                rest_size = rest_size - Reg::i32().size;
            }

            arg.cast_to(CastTarget {
                prefix,
                rest: Uniform { unit: Reg::i64(), total: rest_size },
                attrs: ArgAttributes {
                    regular: arg_attribute,
                    arg_ext: ArgExtension::None,
                    pointee_size: Size::ZERO,
                    pointee_align: None,
                },
            });
            return;
        }
    }

    let total = arg.layout.size;
    if total > in_registers_max {
        arg.make_indirect();
        return;
    }

    arg.cast_to(Uniform { unit: Reg::i64(), total });
}

pub fn compute_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if !fn_abi.ret.is_ignore() {
        classify_arg(cx, &mut fn_abi.ret, Size { raw: 32 });
    }

    for arg in &mut fn_abi.args {
        if arg.is_ignore() {
            continue;
        }
        classify_arg(cx, arg, Size { raw: 16 });
    }
}
