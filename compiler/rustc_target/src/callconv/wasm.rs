use crate::abi::call::{ArgAbi, FnAbi};
use crate::abi::{HasDataLayout, TyAbiInterface};

fn unwrap_trivial_aggregate<'a, Ty, C>(cx: &C, val: &mut ArgAbi<'a, Ty>) -> bool
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if val.layout.is_aggregate() {
        if let Some(unit) = val.layout.homogeneous_aggregate(cx).ok().and_then(|ha| ha.unit()) {
            let size = val.layout.size;
            if unit.size == size {
                val.cast_to(unit);
                return true;
            }
        }
    }
    false
}

fn classify_ret<'a, Ty, C>(cx: &C, ret: &mut ArgAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    ret.extend_integer_width_to(32);
    if ret.layout.is_aggregate() && !unwrap_trivial_aggregate(cx, ret) {
        ret.make_indirect();
    }
}

fn classify_arg<'a, Ty, C>(cx: &C, arg: &mut ArgAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if !arg.layout.is_sized() {
        // Not touching this...
        return;
    }
    arg.extend_integer_width_to(32);
    if arg.layout.is_aggregate() && !unwrap_trivial_aggregate(cx, arg) {
        arg.make_indirect();
    }
}

/// The purpose of this ABI is to match the C ABI (aka clang) exactly.
pub(crate) fn compute_c_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if !fn_abi.ret.is_ignore() {
        classify_ret(cx, &mut fn_abi.ret);
    }

    for arg in fn_abi.args.iter_mut() {
        if arg.is_ignore() {
            continue;
        }
        classify_arg(cx, arg);
    }
}

/// The purpose of this ABI is for matching the WebAssembly standard. This
/// intentionally diverges from the C ABI and is specifically crafted to take
/// advantage of LLVM's support of multiple returns in WebAssembly.
///
/// This ABI is *bad*! It uses `PassMode::Direct` for `abi::Aggregate` types, which leaks LLVM
/// implementation details into the ABI. It's just hard to fix because ABIs are hard to change.
/// Also see <https://github.com/rust-lang/rust/issues/115666>.
pub(crate) fn compute_wasm_abi_info<Ty>(fn_abi: &mut FnAbi<'_, Ty>) {
    if !fn_abi.ret.is_ignore() {
        classify_ret_wasm_abi(&mut fn_abi.ret);
    }

    for arg in fn_abi.args.iter_mut() {
        if arg.is_ignore() {
            continue;
        }
        classify_arg_wasm_abi(arg);
    }

    fn classify_ret_wasm_abi<Ty>(ret: &mut ArgAbi<'_, Ty>) {
        if !ret.layout.is_sized() {
            // Not touching this...
            return;
        }
        // FIXME: this is bad! https://github.com/rust-lang/rust/issues/115666
        ret.make_direct_deprecated();
        ret.extend_integer_width_to(32);
    }

    fn classify_arg_wasm_abi<Ty>(arg: &mut ArgAbi<'_, Ty>) {
        if !arg.layout.is_sized() {
            // Not touching this...
            return;
        }
        // FIXME: this is bad! https://github.com/rust-lang/rust/issues/115666
        arg.make_direct_deprecated();
        arg.extend_integer_width_to(32);
    }
}
