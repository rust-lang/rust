use crate::abi::call::{ArgType, FnType, };
use crate::abi::{HasDataLayout, LayoutOf, TyLayout, TyLayoutMethods};

fn classify_ret_ty<'a, Ty, C>(_cx: &C, ret: &mut ArgType<'a, Ty>)
  where Ty: TyLayoutMethods<'a, C> + Copy,
        C: LayoutOf<Ty = Ty, TyLayout = TyLayout<'a, Ty>> + HasDataLayout
{
  ret.extend_integer_width_to(32);
}

fn classify_arg_ty<'a, Ty, C>(_cx: &C, arg: &mut ArgType<'a, Ty>)
  where Ty: TyLayoutMethods<'a, C> + Copy,
        C: LayoutOf<Ty = Ty, TyLayout = TyLayout<'a, Ty>> + HasDataLayout
{
  arg.extend_integer_width_to(32);
}

pub fn compute_abi_info<'a, Ty, C>(cx: &C, fty: &mut FnType<'a, Ty>)
  where Ty: TyLayoutMethods<'a, C> + Copy,
        C: LayoutOf<Ty = Ty, TyLayout = TyLayout<'a, Ty>> + HasDataLayout
{
  if !fty.ret.is_ignore() {
    classify_ret_ty(cx, &mut fty.ret);
  }

  for arg in &mut fty.args {
    if arg.is_ignore() {
      continue;
    }
    classify_arg_ty(cx, arg);
  }
}
