use arrayvec::ArrayVec;
use rustc_abi::{CanonAbi, HasDataLayout, Reg, Size, TyAbiInterface};

use super::CastTarget;
use crate::callconv::{ArgAbi, ComputeCallConv, Uniform};

fn classify_ret<Ty>(ret: &mut ArgAbi<'_, Ty>) {
    if ret.layout.is_aggregate() && ret.layout.is_sized() {
        classify_aggregate(ret)
    } else if ret.layout.size.bits() < 32 && ret.layout.is_sized() {
        ret.extend_integer_width_to(32);
    }
}

fn classify_arg<'a, Ty, C>(cx: &C, arg: &mut ArgAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
{
    if arg.layout.pass_indirectly_in_non_rustic_abis(cx) {
        arg.make_indirect();
        return;
    }
    if arg.layout.is_aggregate() && arg.layout.is_sized() {
        classify_aggregate(arg)
    } else if arg.layout.size.bits() < 32 && arg.layout.is_sized() {
        arg.extend_integer_width_to(32);
    }
}

/// the pass mode used for aggregates in arg and ret position
fn classify_aggregate<Ty>(arg: &mut ArgAbi<'_, Ty>) {
    let align_bytes = arg.layout.align.bytes();
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
        let mut prefix = ArrayVec::new();
        prefix.push(reg);
        arg.cast_to(CastTarget::prefixed(prefix, Uniform::new(Reg::i8(), Size::ZERO)));
    } else {
        arg.cast_to(Uniform::new(reg, size));
    }
}

fn classify_arg_kernel<'a, Ty, C>(_cx: &C, arg: &mut ArgAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    match arg.mode {
        super::PassMode::Ignore | super::PassMode::Direct(_) => return,
        super::PassMode::Pair(_, _) => {}
        super::PassMode::Cast { .. } | super::PassMode::IndirectUnsized { .. } => unreachable!(),
        super::PassMode::Indirect { .. } => {}
    }

    // FIXME only allow structs and wide pointers here
    // panic!(
    //     "`extern \"ptx-kernel\"` doesn't allow passing types other than primitives and structs"
    // );

    let align_bytes = arg.layout.align.bytes();

    let unit = match align_bytes {
        1 => Reg::i8(),
        2 => Reg::i16(),
        4 => Reg::i32(),
        8 => Reg::i64(),
        16 => Reg::i128(),
        _ => unreachable!("Align is given as power of 2 no larger than 16 bytes"),
    };
    if arg.layout.size.bytes() / align_bytes == 1 {
        // Make sure we pass the struct as array at the LLVM IR level and not as a single integer.
        let mut prefix = ArrayVec::new();
        prefix.push(unit);
        arg.cast_to(CastTarget::prefixed(prefix, Uniform::new(unit, Size::ZERO)));
    } else {
        arg.cast_to(Uniform::new(unit, arg.layout.size));
    }
}

pub(crate) struct Nvptx64CallConv;

impl<'a, Ty, C> ComputeCallConv<'a, Ty, C> for Nvptx64CallConv
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    fn new(_cx: &C) -> Self {
        Self
    }

    fn compute_ret_abi(&mut self, _cx: &C, _conv: CanonAbi, ret: &mut ArgAbi<'a, Ty>) {
        if !ret.is_ignore() {
            classify_ret(ret);
        }
    }

    fn compute_arg_abi(
        &mut self,
        cx: &C,
        _conv: CanonAbi,
        arg: &mut ArgAbi<'a, Ty>,
        _is_vararg: bool,
    ) {
        if arg.is_ignore() {
            return;
        }
        classify_arg(cx, arg);
    }
}

pub(crate) struct Nvptx64KernelCallConv;

impl<'a, Ty, C> ComputeCallConv<'a, Ty, C> for Nvptx64KernelCallConv
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    fn new(_cx: &C) -> Self {
        Self
    }

    fn compute_ret_abi(&mut self, _cx: &C, _conv: CanonAbi, ret: &mut ArgAbi<'a, Ty>) {
        if !ret.layout.is_unit() && !ret.layout.is_never() {
            panic!("Kernels should not return anything other than () or !");
        }
    }

    fn compute_arg_abi(
        &mut self,
        cx: &C,
        _conv: CanonAbi,
        arg: &mut ArgAbi<'a, Ty>,
        _is_vararg: bool,
    ) {
        if arg.is_ignore() {
            return;
        }
        classify_arg_kernel(cx, arg);
    }
}
