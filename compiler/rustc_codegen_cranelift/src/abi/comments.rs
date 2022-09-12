//! Annotate the clif ir with comments describing how arguments are passed into the current function
//! and where all locals are stored.

use std::borrow::Cow;

use rustc_middle::mir;
use rustc_target::abi::call::PassMode;

use cranelift_codegen::entity::EntityRef;

use crate::prelude::*;

pub(super) fn add_args_header_comment(fx: &mut FunctionCx<'_, '_, '_>) {
    if fx.clif_comments.enabled() {
        fx.add_global_comment(
            "kind  loc.idx   param    pass mode                            ty".to_string(),
        );
    }
}

pub(super) fn add_arg_comment<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    kind: &str,
    local: Option<mir::Local>,
    local_field: Option<usize>,
    params: &[Value],
    arg_abi_mode: &PassMode,
    arg_layout: TyAndLayout<'tcx>,
) {
    if !fx.clif_comments.enabled() {
        return;
    }

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
        [] => Cow::Borrowed("-"),
        [param] => Cow::Owned(format!("= {:?}", param)),
        [param_a, param_b] => Cow::Owned(format!("= {:?},{:?}", param_a, param_b)),
        params => Cow::Owned(format!(
            "= {}",
            params.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
        )),
    };

    let pass_mode = format!("{:?}", arg_abi_mode);
    fx.add_global_comment(format!(
        "{kind:5}{local:>3}{local_field:<5} {params:10} {pass_mode:36} {ty:?}",
        kind = kind,
        local = local,
        local_field = local_field,
        params = params,
        pass_mode = pass_mode,
        ty = arg_layout.ty,
    ));
}

pub(super) fn add_locals_header_comment(fx: &mut FunctionCx<'_, '_, '_>) {
    if fx.clif_comments.enabled() {
        fx.add_global_comment(String::new());
        fx.add_global_comment(
            "kind  local ty                              size align (abi,pref)".to_string(),
        );
    }
}

pub(super) fn add_local_place_comments<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    place: CPlace<'tcx>,
    local: Local,
) {
    if !fx.clif_comments.enabled() {
        return;
    }
    let TyAndLayout { ty, layout } = place.layout();
    let rustc_target::abi::LayoutS {
        size,
        align,
        abi: _,
        variants: _,
        fields: _,
        largest_niche: _,
    } = layout.0.0;

    let (kind, extra) = match *place.inner() {
        CPlaceInner::Var(place_local, var) => {
            assert_eq!(local, place_local);
            ("ssa", Cow::Owned(format!(",var={}", var.index())))
        }
        CPlaceInner::VarPair(place_local, var1, var2) => {
            assert_eq!(local, place_local);
            ("ssa", Cow::Owned(format!(",var=({}, {})", var1.index(), var2.index())))
        }
        CPlaceInner::VarLane(_local, _var, _lane) => unreachable!(),
        CPlaceInner::Addr(ptr, meta) => {
            let meta = if let Some(meta) = meta {
                Cow::Owned(format!(",meta={}", meta))
            } else {
                Cow::Borrowed("")
            };
            match ptr.debug_base_and_offset() {
                (crate::pointer::PointerBase::Addr(addr), offset) => {
                    ("reuse", format!("storage={}{}{}", addr, offset, meta).into())
                }
                (crate::pointer::PointerBase::Stack(stack_slot), offset) => {
                    ("stack", format!("storage={}{}{}", stack_slot, offset, meta).into())
                }
                (crate::pointer::PointerBase::Dangling(align), offset) => {
                    ("zst", format!("align={},offset={}", align.bytes(), offset).into())
                }
            }
        }
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
