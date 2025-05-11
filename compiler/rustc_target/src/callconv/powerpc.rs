use crate::callconv::{ArgAbi, FnAbi};
use crate::spec::HasTargetSpec;

fn classify_ret<Ty>(ret: &mut ArgAbi<'_, Ty>) {
    if ret.layout.is_aggregate() {
        ret.make_indirect();
    } else {
        ret.extend_integer_width_to(32);
    }
}

fn classify_arg<Ty>(cx: &impl HasTargetSpec, arg: &mut ArgAbi<'_, Ty>) {
    if arg.is_ignore() {
        // powerpc-unknown-linux-{gnu,musl,uclibc} doesn't ignore ZSTs.
        if cx.target_spec().os == "linux"
            && matches!(&*cx.target_spec().env, "gnu" | "musl" | "uclibc")
            && arg.layout.is_zst()
        {
            arg.make_indirect_from_ignore();
        }
        return;
    }
    if arg.layout.is_aggregate() {
        arg.make_indirect();
    } else {
        arg.extend_integer_width_to(32);
    }
}

pub(crate) fn compute_abi_info<Ty>(cx: &impl HasTargetSpec, fn_abi: &mut FnAbi<'_, Ty>) {
    if !fn_abi.ret.is_ignore() {
        classify_ret(&mut fn_abi.ret);
    }

    for arg in fn_abi.args.iter_mut() {
        classify_arg(cx, arg);
    }
}
