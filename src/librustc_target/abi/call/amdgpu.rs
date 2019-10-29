use crate::abi::call::{ArgAbi, FnAbi, };
use crate::abi::{HasDataLayout, LayoutOf, TyLayout, TyLayoutMethods};

fn classify_ret<'a, Ty, C>(_cx: &C, ret: &mut ArgAbi<'a, Ty>)
  where Ty: TyLayoutMethods<'a, C> + Copy,
        C: LayoutOf<Ty = Ty, TyLayout = TyLayout<'a, Ty>> + HasDataLayout
{
  ret.extend_integer_width_to(32);
}

fn classify_arg<'a, Ty, C>(_cx: &C, arg: &mut ArgAbi<'a, Ty>)
  where Ty: TyLayoutMethods<'a, C> + Copy,
        C: LayoutOf<Ty = Ty, TyLayout = TyLayout<'a, Ty>> + HasDataLayout
{
  arg.extend_integer_width_to(32);
}

pub fn compute_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>)
  where Ty: TyLayoutMethods<'a, C> + Copy,
        C: LayoutOf<Ty = Ty, TyLayout = TyLayout<'a, Ty>> + HasDataLayout
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
