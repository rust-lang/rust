// FIXME: This needs an audit for correctness and completeness.

use crate::abi::call::{
    ArgAbi, ArgAttribute, ArgAttributes, ArgExtension, CastTarget, FnAbi, Reg, Uniform,
};
use crate::abi::{self, HasDataLayout, Scalar, Size, TyAbiInterface, TyAndLayout};

#[derive(Clone, Debug)]
pub struct Sdata {
    pub prefix: [Option<Reg>; 8],
    pub prefix_index: usize,
    pub last_offset: Size,
    pub has_float: bool,
    pub arg_attribute: ArgAttribute,
}

fn arg_scalar<C>(cx: &C, scalar: &Scalar, offset: Size, mut data: Sdata) -> Sdata
where
    C: HasDataLayout,
{
    let dl = cx.data_layout();

    if !scalar.primitive().is_float() {
        return data;
    }

    data.has_float = true;

    if !data.last_offset.is_aligned(dl.f64_align.abi) && data.last_offset < offset {
        if data.prefix_index == data.prefix.len() {
            return data;
        }
        data.prefix[data.prefix_index] = Some(Reg::i32());
        data.prefix_index += 1;
        data.last_offset = data.last_offset + Reg::i32().size;
    }

    for _ in 0..((offset - data.last_offset).bits() / 64)
        .min((data.prefix.len() - data.prefix_index) as u64)
    {
        data.prefix[data.prefix_index] = Some(Reg::i64());
        data.prefix_index += 1;
        data.last_offset = data.last_offset + Reg::i64().size;
    }

    if data.last_offset < offset {
        if data.prefix_index == data.prefix.len() {
            return data;
        }
        data.prefix[data.prefix_index] = Some(Reg::i32());
        data.prefix_index += 1;
        data.last_offset = data.last_offset + Reg::i32().size;
    }

    if data.prefix_index == data.prefix.len() {
        return data;
    }

    if scalar.primitive() == abi::F32 {
        data.arg_attribute = ArgAttribute::InReg;
        data.prefix[data.prefix_index] = Some(Reg::f32());
        data.last_offset = offset + Reg::f32().size;
    } else {
        data.prefix[data.prefix_index] = Some(Reg::f64());
        data.last_offset = offset + Reg::f64().size;
    }
    data.prefix_index += 1;
    return data;
}

fn arg_scalar_pair<C>(
    cx: &C,
    scalar1: &Scalar,
    scalar2: &Scalar,
    mut offset: Size,
    mut data: Sdata,
) -> Sdata
where
    C: HasDataLayout,
{
    data = arg_scalar(cx, &scalar1, offset, data);
    match (scalar1.primitive(), scalar2.primitive()) {
        (abi::F32, _) => offset += Reg::f32().size,
        (_, abi::F64) => offset += Reg::f64().size,
        (abi::Int(i, _signed), _) => offset += i.size(),
        (abi::Pointer, _) => offset += Reg::i64().size,
        _ => {}
    }

    if (offset.raw % 4) != 0 && scalar2.primitive().is_float() {
        offset.raw += 4 - (offset.raw % 4);
    }
    data = arg_scalar(cx, &scalar2, offset, data);
    return data;
}

fn parse_structure<'a, Ty, C>(
    cx: &C,
    layout: TyAndLayout<'a, Ty>,
    mut data: Sdata,
    mut offset: Size,
) -> Sdata
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if let abi::FieldsShape::Union(_) = layout.fields {
        return data;
    }

    match layout.abi {
        abi::Abi::Scalar(scalar) => {
            data = arg_scalar(cx, &scalar, offset, data);
        }
        abi::Abi::Aggregate { .. } => {
            for i in 0..layout.fields.count() {
                if offset < layout.fields.offset(i) {
                    offset = layout.fields.offset(i);
                }
                data = parse_structure(cx, layout.field(cx, i), data.clone(), offset);
            }
        }
        _ => {
            if let abi::Abi::ScalarPair(scalar1, scalar2) = &layout.abi {
                data = arg_scalar_pair(cx, scalar1, scalar2, offset, data);
            }
        }
    }

    return data;
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

    let total = arg.layout.size;
    if total > in_registers_max {
        arg.make_indirect();
        return;
    }

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
            // Structures with floating point numbers need special care.

            let mut data = parse_structure(
                cx,
                arg.layout,
                Sdata {
                    prefix: [None; 8],
                    prefix_index: 0,
                    last_offset: Size::ZERO,
                    has_float: false,
                    arg_attribute: ArgAttribute::default(),
                },
                Size { raw: 0 },
            );

            if data.has_float {
                // Structure { float, int, int } doesn't like to be handled like
                // { float, long int }. Other way around it doesn't mind.
                if data.last_offset < arg.layout.size
                    && (data.last_offset.raw % 8) != 0
                    && data.prefix_index < data.prefix.len()
                {
                    data.prefix[data.prefix_index] = Some(Reg::i32());
                    data.prefix_index += 1;
                    data.last_offset += Reg::i32().size;
                }

                let mut rest_size = arg.layout.size - data.last_offset;
                if (rest_size.raw % 8) != 0 && data.prefix_index < data.prefix.len() {
                    data.prefix[data.prefix_index] = Some(Reg::i32());
                    rest_size = rest_size - Reg::i32().size;
                }

                arg.cast_to(CastTarget {
                    prefix: data.prefix,
                    rest: Uniform { unit: Reg::i64(), total: rest_size },
                    attrs: ArgAttributes {
                        regular: data.arg_attribute,
                        arg_ext: ArgExtension::None,
                        pointee_size: Size::ZERO,
                        pointee_align: None,
                    },
                });
                return;
            }
        }
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

    for arg in fn_abi.args.iter_mut() {
        if arg.is_ignore() {
            continue;
        }
        classify_arg(cx, arg, Size { raw: 16 });
    }
}
