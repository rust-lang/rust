use std::borrow::Cow;

use rustc::mir;

use crate::abi::pass_mode::*;
use crate::prelude::*;

pub fn add_args_header_comment(fx: &mut FunctionCx<impl Backend>) {
    fx.add_global_comment(format!(
        "kind  loc.idx   param    pass mode                            ty"
    ));
}

pub fn add_arg_comment<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
    kind: &str,
    local: Option<mir::Local>,
    local_field: Option<usize>,
    params: EmptySinglePair<Value>,
    pass_mode: PassMode,
    ty: Ty<'tcx>,
) {
    let local = if let Some(local) = local {
        Cow::Owned(format!("{:?}", local))
    } else {
        Cow::Borrowed("???")
    };
    let local_field = if let Some(local_field) = local_field {
        Cow::Owned(format!(".{}", local_field))
    } else {
        Cow::Borrowed("")
    };

    let params = match params {
        Empty => Cow::Borrowed("-"),
        Single(param) => Cow::Owned(format!("= {:?}", param)),
        Pair(param_a, param_b) => Cow::Owned(format!("= {:?}, {:?}", param_a, param_b)),
    };

    let pass_mode = format!("{:?}", pass_mode);
    fx.add_global_comment(format!(
        "{kind:5}{local:>3}{local_field:<5} {params:10} {pass_mode:36} {ty:?}",
        kind = kind,
        local = local,
        local_field = local_field,
        params = params,
        pass_mode = pass_mode,
        ty = ty,
    ));
}

pub fn add_locals_header_comment(fx: &mut FunctionCx<impl Backend>) {
    fx.add_global_comment(String::new());
    fx.add_global_comment(format!(
        "kind  local ty                              size align (abi,pref)"
    ));
}

pub fn add_local_place_comments<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
    place: CPlace<'tcx>,
    local: Local,
) {
    let TyLayout { ty, details } = place.layout();
    let ty::layout::LayoutDetails {
        size,
        align,
        abi: _,
        variants: _,
        fields: _,
        largest_niche: _,
    } = details;

    let (kind, extra) = match *place.inner() {
        CPlaceInner::Var(var) => {
            assert_eq!(local, var);
            ("ssa", std::borrow::Cow::Borrowed(""))
        }
        CPlaceInner::NoPlace => ("zst", "".into()),
        CPlaceInner::Addr(ptr, None) => match ptr.base_and_offset() {
            (crate::pointer::PointerBase::Addr(addr), offset) => {
                ("reuse", format!("storage={}{}", addr, offset).into())
            }
            (crate::pointer::PointerBase::Stack(stack_slot), offset) => {
                ("stack", format!("storage={}{}", stack_slot, offset).into())
            }
        },
        CPlaceInner::Addr(_, Some(_)) => unreachable!(),
    };

    fx.add_global_comment(format!(
        "{:<5} {:5} {:30} {:4}b {}, {}{}{}",
        kind,
        format!("{:?}", local),
        format!("{:?}", ty),
        size.bytes(),
        align.abi.bytes(),
        align.pref.bytes(),
        if extra.is_empty() { "" } else { "              " },
        extra,
    ));
}
