use std::borrow::Cow;

use rustc::mir;

use crate::prelude::*;
use crate::abi::pass_mode::*;

pub fn add_local_header_comment(fx: &mut FunctionCx<impl Backend>) {
    fx.add_global_comment(format!(
        "msg   loc.idx    param    pass mode                            ssa flags  ty"
    ));
}

pub fn add_arg_comment<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
    msg: &str,
    local: mir::Local,
    local_field: Option<usize>,
    params: EmptySinglePair<Value>,
    pass_mode: PassMode,
    ssa: crate::analyze::Flags,
    ty: Ty<'tcx>,
) {
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
        "{msg:5}{local:>3}{local_field:<5} {params:10} {pass_mode:36} {ssa:10} {ty:?}",
        msg = msg,
        local = format!("{:?}", local),
        local_field = local_field,
        params = params,
        pass_mode = pass_mode,
        ssa = format!("{:?}", ssa),
        ty = ty,
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
    match *place.inner() {
        CPlaceInner::Var(var) => {
            assert_eq!(local, var);
            fx.add_global_comment(format!(
                "ssa   {:?}: {:?} size={} align={}, {}",
                local,
                ty,
                size.bytes(),
                align.abi.bytes(),
                align.pref.bytes(),
            ));
        }
        CPlaceInner::Stack(stack_slot) => fx.add_entity_comment(
            stack_slot,
            format!(
                "{:?}: {:?} size={} align={},{}",
                local,
                ty,
                size.bytes(),
                align.abi.bytes(),
                align.pref.bytes(),
            ),
        ),
        CPlaceInner::NoPlace => fx.add_global_comment(format!(
            "zst   {:?}: {:?} size={} align={}, {}",
            local,
            ty,
            size.bytes(),
            align.abi.bytes(),
            align.pref.bytes(),
        )),
        CPlaceInner::Addr(_, _) => unreachable!(),
    }
}
