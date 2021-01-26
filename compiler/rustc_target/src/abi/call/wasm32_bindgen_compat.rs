// This is not and has never been a correct C ABI for WebAssembly, but
// for a long time this was the C ABI that Rust used. wasm-bindgen
// depends on ABI details for this ABI and is incompatible with the
// correct C ABI, so this ABI is being kept around until wasm-bindgen
// can be fixed to work with the correct ABI. See #63649 for further
// discussion.

use rustc_data_structures::stable_set::FxHashSet;
use rustc_span::Symbol;

use crate::abi::call::{ArgAbi, FnAbi, Uniform};
use crate::abi::{HasDataLayout, LayoutOf, TyAndLayout, TyAndLayoutMethods};

fn classify_ret<'a, Ty, C>(cx: &C, target_features: &FxHashSet<Symbol>, ret: &mut ArgAbi<'a, Ty>)
where
    Ty: TyAndLayoutMethods<'a, C> + Copy,
    C: LayoutOf<Ty = Ty, TyAndLayout = TyAndLayout<'a, Ty>> + HasDataLayout,
{
    if ret.layout.is_aggregate() {
        if let Some(unit) = ret.layout.homogeneous_aggregate(cx).ok().and_then(|ha| ha.unit()) {
            let size = ret.layout.size;
            if unit.size == size || target_features.contains(&Symbol::intern("multivalue")) {
                ret.cast_to(Uniform { unit, total: size });
            }
        }
    }
    ret.extend_integer_width_to(32);
}

fn classify_arg<'a, Ty, C>(cx: &C, arg: &mut ArgAbi<'a, Ty>)
where
    Ty: TyAndLayoutMethods<'a, C> + Copy,
    C: LayoutOf<Ty = Ty, TyAndLayout = TyAndLayout<'a, Ty>> + HasDataLayout,
{
    if arg.layout.is_aggregate() {
        if let Some(unit) = arg.layout.homogeneous_aggregate(cx).ok().and_then(|ha| ha.unit()) {
            let size = arg.layout.size;
            arg.cast_to(Uniform { unit, total: size });
        }
    }
    arg.extend_integer_width_to(32);
}

pub fn compute_abi_info<'a, Ty, C>(
    cx: &C,
    target_features: &FxHashSet<Symbol>,
    fn_abi: &mut FnAbi<'a, Ty>,
) where
    Ty: TyAndLayoutMethods<'a, C> + Copy,
    C: LayoutOf<Ty = Ty, TyAndLayout = TyAndLayout<'a, Ty>> + HasDataLayout,
{
    if !fn_abi.ret.is_ignore() {
        classify_ret(cx, target_features, &mut fn_abi.ret);
    }

    for arg in &mut fn_abi.args {
        if arg.is_ignore() {
            continue;
        }
        classify_arg(cx, arg);
    }
}
