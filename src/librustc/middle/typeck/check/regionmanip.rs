// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// #[warn(deprecated_mode)];

use core::prelude::*;

use middle::ty::{FnTyBase};
use middle::ty;
use middle::typeck::check::self_info;
use middle::typeck::isr_alist;
use util::common::indenter;
use util::ppaux::region_to_str;
use util::ppaux;

use std::list::Cons;
use syntax::ast;
use syntax::print::pprust::{expr_to_str};

// Helper functions related to manipulating region types.

pub fn replace_bound_regions_in_fn_ty(
    tcx: ty::ctxt,
    isr: isr_alist,
    self_info: Option<self_info>,
    fn_ty: &ty::FnTy,
    mapf: fn(ty::bound_region) -> ty::Region) ->
    {isr: isr_alist, self_info: Option<self_info>, fn_ty: ty::FnTy}
{
    let {isr, self_info, fn_sig} =
        replace_bound_regions_in_fn_sig(
            tcx, isr, self_info, &fn_ty.sig, mapf);
    {isr: isr,
     self_info: self_info,
     fn_ty: FnTyBase {meta: fn_ty.meta,
                      sig: fn_sig}}
}

pub fn replace_bound_regions_in_fn_sig(
    tcx: ty::ctxt,
    isr: isr_alist,
    self_info: Option<self_info>,
    fn_sig: &ty::FnSig,
    mapf: fn(ty::bound_region) -> ty::Region) ->
    {isr: isr_alist, self_info: Option<self_info>, fn_sig: ty::FnSig}
{
    // Take self_info apart; the self_ty part is the only one we want
    // to update here.
    let self_ty = self_info.map(|s| s.self_ty);
    let rebuild_self_info = |t| self_info.map(|s| {self_ty: t, ..*s});

    let mut all_tys = ty::tys_in_fn_sig(fn_sig);

    match self_info {
      Some({explicit_self: ast::spanned { node: ast::sty_region(m),
                                          _}, _}) => {
        let region = ty::re_bound(ty::br_self);
        let ty = ty::mk_rptr(tcx, region,
                             ty::mt { ty: ty::mk_self(tcx), mutbl: m });
        all_tys.push(ty);
      }
      _ => {}
    }


    for self_ty.each |t| { all_tys.push(*t) }

    debug!("replace_bound_regions_in_fn_sig(self_info.self_ty=%?, fn_sig=%s, \
            all_tys=%?)",
           self_ty.map(|t| ppaux::ty_to_str(tcx, *t)),
           ppaux::fn_sig_to_str(tcx, fn_sig),
           all_tys.map(|t| ppaux::ty_to_str(tcx, *t)));
    let _i = indenter();

    let isr = do create_bound_region_mapping(tcx, isr, all_tys) |br| {
        debug!("br=%?", br);
        mapf(br)
    };
    let new_fn_sig = ty::fold_sig(fn_sig, |t| {
        replace_bound_regions(tcx, isr, t)
    });
    let t_self = self_ty.map(|t| replace_bound_regions(tcx, isr, *t));

    debug!("result of replace_bound_regions_in_fn_sig: self_info.self_ty=%?, \
                fn_sig=%s",
           t_self.map(|t| ppaux::ty_to_str(tcx, *t)),
           ppaux::fn_sig_to_str(tcx, &new_fn_sig));

    // Glue updated self_ty back together with its original def_id.
    let new_self_info: Option<self_info> = match t_self {
      None    => None,
      Some(t) => rebuild_self_info(t)
    };

    return {isr: isr,
            self_info: new_self_info,
            fn_sig: new_fn_sig};

    // Takes `isr`, a (possibly empty) mapping from in-scope region
    // names ("isr"s) to their corresponding regions; `tys`, a list of
    // types, and `to_r`, a closure that takes a bound_region and
    // returns a region.  Returns an updated version of `isr`,
    // extended with the in-scope region names from all of the bound
    // regions appearing in the types in the `tys` list (if they're
    // not in `isr` already), with each of those in-scope region names
    // mapped to a region that's the result of applying `to_r` to
    // itself.
    fn create_bound_region_mapping(
        tcx: ty::ctxt,
        isr: isr_alist,
        tys: ~[ty::t],
        to_r: fn(ty::bound_region) -> ty::Region) -> isr_alist {

        // Takes `isr` (described above), `to_r` (described above),
        // and `r`, a region.  If `r` is anything other than a bound
        // region, or if it's a bound region that already appears in
        // `isr`, then we return `isr` unchanged.  If `r` is a bound
        // region that doesn't already appear in `isr`, we return an
        // updated isr_alist that now contains a mapping from `r` to
        // the result of calling `to_r` on it.
        fn append_isr(isr: isr_alist,
                      to_r: fn(ty::bound_region) -> ty::Region,
                      r: ty::Region) -> isr_alist {
            match r {
              ty::re_free(_, _) | ty::re_static | ty::re_scope(_) |
              ty::re_infer(_) => {
                isr
              }
              ty::re_bound(br) => {
                match isr.find(br) {
                  Some(_) => isr,
                  None => @Cons((br, to_r(br)), isr)
                }
              }
            }
        }

        // For each type `ty` in `tys`...
        do tys.foldl(isr) |isr, ty| {
            let mut isr = *isr;

            // Using fold_regions is inefficient, because it
            // constructs new types, but it avoids code duplication in
            // terms of locating all the regions within the various
            // kinds of types.  This had already caused me several
            // bugs so I decided to switch over.
            do ty::fold_regions(tcx, *ty) |r, in_fn| {
                if !in_fn { isr = append_isr(isr, to_r, r); }
                r
            };

            isr
        }
    }

    // Takes `isr`, a mapping from in-scope region names ("isr"s) to
    // their corresponding regions; and `ty`, a type.  Returns an
    // updated version of `ty`, in which bound regions in `ty` have
    // been replaced with the corresponding bindings in `isr`.
    fn replace_bound_regions(
        tcx: ty::ctxt,
        isr: isr_alist,
        ty: ty::t) -> ty::t {

        do ty::fold_regions(tcx, ty) |r, in_fn| {
            let r1 = match r {
              // As long as we are not within a fn() type, `&T` is
              // mapped to the free region anon_r.  But within a fn
              // type, it remains bound.
              ty::re_bound(ty::br_anon(_)) if in_fn => r,

              ty::re_bound(br) => {
                match isr.find(br) {
                  // In most cases, all named, bound regions will be
                  // mapped to some free region.
                  Some(fr) => fr,

                  // But in the case of a fn() type, there may be
                  // named regions within that remain bound:
                  None if in_fn => r,
                  None => {
                    tcx.sess.bug(
                        fmt!("Bound region not found in \
                              in_scope_regions list: %s",
                             region_to_str(tcx, r)));
                  }
                }
              }

              // Free regions like these just stay the same:
              ty::re_static |
              ty::re_scope(_) |
              ty::re_free(_, _) |
              ty::re_infer(_) => r
            };
            r1
        }
    }
}
