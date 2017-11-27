// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use rustc::ty::{self, TyCtxt};
use rustc::mir::*;

/// Return `true` if this lvalue is allowed to be less aligned
/// than its containing struct (because it is within a packed
/// struct).
pub fn is_disaligned<'a, 'tcx, L>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                  local_decls: &L,
                                  param_env: ty::ParamEnv<'tcx>,
                                  lvalue: &Lvalue<'tcx>)
                                  -> bool
    where L: HasLocalDecls<'tcx>
{
    debug!("is_disaligned({:?})", lvalue);
    if !is_within_packed(tcx, local_decls, lvalue) {
        debug!("is_disaligned({:?}) - not within packed", lvalue);
        return false
    }

    let ty = lvalue.ty(local_decls, tcx).to_ty(tcx);
    match tcx.layout_raw(param_env.and(ty)) {
        Ok(layout) if layout.align.abi() == 1 => {
            // if the alignment is 1, the type can't be further
            // disaligned.
            debug!("is_disaligned({:?}) - align = 1", lvalue);
            false
        }
        _ => {
            debug!("is_disaligned({:?}) - true", lvalue);
            true
        }
    }
}

fn is_within_packed<'a, 'tcx, L>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                 local_decls: &L,
                                 lvalue: &Lvalue<'tcx>)
                                 -> bool
    where L: HasLocalDecls<'tcx>
{
    let mut lvalue = lvalue;
    while let &Lvalue::Projection(box Projection {
        ref base, ref elem
    }) = lvalue {
        match *elem {
            // encountered a Deref, which is ABI-aligned
            ProjectionElem::Deref => break,
            ProjectionElem::Field(..) => {
                let ty = base.ty(local_decls, tcx).to_ty(tcx);
                match ty.sty {
                    ty::TyAdt(def, _) if def.repr.packed() => {
                        return true
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        lvalue = base;
    }

    false
}
