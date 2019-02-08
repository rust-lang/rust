use crate::abi::call::{FnType, ArgType, Reg, RegKind, Uniform};
use crate::abi::{HasDataLayout, LayoutOf, TyLayout, TyLayoutMethods};

fn is_homogeneous_aggregate<'a, Ty, C>(cx: &C, arg: &mut ArgType<'a, Ty>)
                                     -> Option<Uniform>
    where Ty: TyLayoutMethods<'a, C> + Copy,
          C: LayoutOf<Ty = Ty, TyLayout = TyLayout<'a, Ty>> + HasDataLayout
{
    arg.layout.homogeneous_aggregate(cx).unit().and_then(|unit| {
        let size = arg.layout.size;

        // Ensure we have at most four uniquely addressable members.
        if size > unit.size.checked_mul(4, cx).unwrap() {
            return None;
        }

        let valid_unit = match unit.kind {
            RegKind::Integer => false,
            RegKind::Float => true,
            RegKind::Vector => size.bits() == 64 || size.bits() == 128
        };

        if valid_unit {
            Some(Uniform {
                unit,
                total: size
            })
        } else {
            None
        }
    })
}

fn classify_ret_ty<'a, Ty, C>(cx: &C, ret: &mut ArgType<'a, Ty>)
    where Ty: TyLayoutMethods<'a, C> + Copy,
          C: LayoutOf<Ty = Ty, TyLayout = TyLayout<'a, Ty>> + HasDataLayout
{
    if !ret.layout.is_aggregate() {
        ret.extend_integer_width_to(32);
        return;
    }
    if let Some(uniform) = is_homogeneous_aggregate(cx, ret) {
        ret.cast_to(uniform);
        return;
    }
    let size = ret.layout.size;
    let bits = size.bits();
    if bits <= 128 {
        let unit = if bits <= 8 {
            Reg::i8()
        } else if bits <= 16 {
            Reg::i16()
        } else if bits <= 32 {
            Reg::i32()
        } else {
            Reg::i64()
        };

        ret.cast_to(Uniform {
            unit,
            total: size
        });
        return;
    }
    ret.make_indirect();
}

fn classify_arg_ty<'a, Ty, C>(cx: &C, arg: &mut ArgType<'a, Ty>)
    where Ty: TyLayoutMethods<'a, C> + Copy,
          C: LayoutOf<Ty = Ty, TyLayout = TyLayout<'a, Ty>> + HasDataLayout
{
    if !arg.layout.is_aggregate() {
        arg.extend_integer_width_to(32);
        return;
    }
    if let Some(uniform) = is_homogeneous_aggregate(cx, arg) {
        arg.cast_to(uniform);
        return;
    }
    let size = arg.layout.size;
    let bits = size.bits();
    if bits <= 128 {
        let unit = if bits <= 8 {
            Reg::i8()
        } else if bits <= 16 {
            Reg::i16()
        } else if bits <= 32 {
            Reg::i32()
        } else {
            Reg::i64()
        };

        arg.cast_to(Uniform {
            unit,
            total: size
        });
        return;
    }
    arg.make_indirect();
}

pub fn compute_abi_info<'a, Ty, C>(cx: &C, fty: &mut FnType<'a, Ty>)
    where Ty: TyLayoutMethods<'a, C> + Copy,
          C: LayoutOf<Ty = Ty, TyLayout = TyLayout<'a, Ty>> + HasDataLayout
{
    if !fty.ret.is_ignore() {
        classify_ret_ty(cx, &mut fty.ret);
    }

    for arg in &mut fty.args {
        if arg.is_ignore() { continue; }
        classify_arg_ty(cx, arg);
    }
}
