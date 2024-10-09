use super::{ArgAttribute, ArgAttributes, ArgExtension, CastTarget};
use crate::abi::call::{ArgAbi, FnAbi, PassMode, Reg, Size, Uniform};
use crate::abi::{HasDataLayout, TyAbiInterface};

fn classify_ret<Ty>(ret: &mut ArgAbi<'_, Ty>) {
    if ret.layout.is_aggregate() && ret.layout.is_sized() {
        classify_aggregate(ret)
    } else if ret.layout.size.bits() < 32 && ret.layout.is_sized() {
        ret.extend_integer_width_to(32);
    }
}

fn classify_arg<Ty>(arg: &mut ArgAbi<'_, Ty>) {
    if arg.layout.is_aggregate() && arg.layout.is_sized() {
        classify_aggregate(arg)
    } else if arg.layout.size.bits() < 32 && arg.layout.is_sized() {
        arg.extend_integer_width_to(32);
    }
}

/// the pass mode used for aggregates in arg and ret position
fn classify_aggregate<Ty>(arg: &mut ArgAbi<'_, Ty>) {
    let align_bytes = arg.layout.align.abi.bytes();
    let size = arg.layout.size;

    let reg = match align_bytes {
        1 => Reg::i8(),
        2 => Reg::i16(),
        4 => Reg::i32(),
        8 => Reg::i64(),
        16 => Reg::i128(),
        _ => unreachable!("Align is given as power of 2 no larger than 16 bytes"),
    };

    if align_bytes == size.bytes() {
        arg.cast_to(CastTarget {
            prefix: [Some(reg), None, None, None, None, None, None, None],
            rest: Uniform::new(Reg::i8(), Size::from_bytes(0)),
            attrs: ArgAttributes {
                regular: ArgAttribute::default(),
                arg_ext: ArgExtension::None,
                pointee_size: Size::ZERO,
                pointee_align: None,
            },
        });
    } else {
        arg.cast_to(Uniform::new(reg, size));
    }
}

fn classify_arg_kernel<'a, Ty, C>(_cx: &C, arg: &mut ArgAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if matches!(arg.mode, PassMode::Pair(..)) && (arg.layout.is_adt() || arg.layout.is_tuple()) {
        let align_bytes = arg.layout.align.abi.bytes();

        let unit = match align_bytes {
            1 => Reg::i8(),
            2 => Reg::i16(),
            4 => Reg::i32(),
            8 => Reg::i64(),
            16 => Reg::i128(),
            _ => unreachable!("Align is given as power of 2 no larger than 16 bytes"),
        };
        arg.cast_to(Uniform::new(unit, Size::from_bytes(2 * align_bytes)));
    } else {
        // FIXME: find a better way to do this. See https://github.com/rust-lang/rust/issues/117271.
        arg.make_direct_deprecated();
    }
}

pub(crate) fn compute_abi_info<Ty>(fn_abi: &mut FnAbi<'_, Ty>) {
    if !fn_abi.ret.is_ignore() {
        classify_ret(&mut fn_abi.ret);
    }

    for arg in fn_abi.args.iter_mut() {
        if arg.is_ignore() {
            continue;
        }
        classify_arg(arg);
    }
}

pub(crate) fn compute_ptx_kernel_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if !fn_abi.ret.layout.is_unit() && !fn_abi.ret.layout.is_never() {
        panic!("Kernels should not return anything other than () or !");
    }

    for arg in fn_abi.args.iter_mut() {
        if arg.is_ignore() {
            continue;
        }
        classify_arg_kernel(cx, arg);
    }
}
