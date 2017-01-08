// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! An experimental pass that scources for `#[rustc_mir]` attributes,
//! builds the resulting MIR, and dumps it out into a file for inspection.
//!
//! The attribute formats that are currently accepted are:
//!
//! - `#[rustc_mir(graphviz="file.gv")]`
//! - `#[rustc_mir(pretty="file.mir")]`

use build;
use rustc::dep_graph::DepNode;
use rustc::mir::Mir;
use rustc::mir::transform::MirSource;
use rustc::mir::visit::MutVisitor;
use pretty;
use hair::cx::Cx;

use rustc::infer::InferCtxt;
use rustc::traits::Reveal;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::subst::Substs;
use rustc::hir;
use rustc::hir::intravisit::{Visitor, NestedVisitorMap};
use syntax::abi::Abi;
use syntax::ast;
use syntax_pos::Span;

use std::mem;

pub fn build_mir_for_crate<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    tcx.visit_all_item_likes_in_krate(DepNode::Mir, &mut BuildMir {
        tcx: tcx
    }.as_deep_visitor());
}

/// A pass to lift all the types and substitutions in a Mir
/// to the global tcx. Sadly, we don't have a "folder" that
/// can change 'tcx so we have to transmute afterwards.
struct GlobalizeMir<'a, 'gcx: 'a> {
    tcx: TyCtxt<'a, 'gcx, 'gcx>,
    span: Span
}

impl<'a, 'gcx: 'tcx, 'tcx> MutVisitor<'tcx> for GlobalizeMir<'a, 'gcx> {
    fn visit_ty(&mut self, ty: &mut Ty<'tcx>) {
        if let Some(lifted) = self.tcx.lift(ty) {
            *ty = lifted;
        } else {
            span_bug!(self.span,
                      "found type `{:?}` with inference types/regions in MIR",
                      ty);
        }
    }

    fn visit_substs(&mut self, substs: &mut &'tcx Substs<'tcx>) {
        if let Some(lifted) = self.tcx.lift(substs) {
            *substs = lifted;
        } else {
            span_bug!(self.span,
                      "found substs `{:?}` with inference types/regions in MIR",
                      substs);
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// BuildMir -- walks a crate, looking for fn items and methods to build MIR from

struct BuildMir<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>
}

fn build<'a, 'gcx, 'tcx>(infcx: &InferCtxt<'a, 'gcx, 'tcx>,
                         body_id: hir::BodyId)
                         -> (Mir<'tcx>, MirSource) {
    let tcx = infcx.tcx.global_tcx();

    let item_id = tcx.map.body_owner(body_id);
    let src = MirSource::from_node(tcx, item_id);
    let cx = Cx::new(infcx, src);
    if let MirSource::Fn(id) = src {
        // fetch the fully liberated fn signature (that is, all bound
        // types/lifetimes replaced)
        let fn_sig = cx.tables().liberated_fn_sigs[&id].clone();

        let ty = tcx.item_type(tcx.map.local_def_id(id));
        let (abi, implicit_argument) = if let ty::TyClosure(..) = ty.sty {
            (Abi::Rust, Some((closure_self_ty(tcx, id, body_id), None)))
        } else {
            (ty.fn_abi(), None)
        };

        let body = tcx.map.body(body_id);
        let explicit_arguments =
            body.arguments
                .iter()
                .enumerate()
                .map(|(index, arg)| {
                    (fn_sig.inputs()[index], Some(&*arg.pat))
                });

        let arguments = implicit_argument.into_iter().chain(explicit_arguments);
        (build::construct_fn(cx, id, arguments, abi, fn_sig.output(), body), src)
    } else {
        (build::construct_const(cx, body_id), src)
    }
}

impl<'a, 'tcx> Visitor<'tcx> for BuildMir<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }

    fn visit_nested_body(&mut self, body_id: hir::BodyId) {
        self.tcx.infer_ctxt(body_id, Reveal::NotSpecializable).enter(|infcx| {
            let (mut mir, src) = build(&infcx, body_id);

            // Convert the Mir to global types.
            let tcx = infcx.tcx.global_tcx();
            let mut globalizer = GlobalizeMir {
                tcx: tcx,
                span: mir.span
            };
            globalizer.visit_mir(&mut mir);
            let mir = unsafe {
                mem::transmute::<Mir, Mir<'tcx>>(mir)
            };

            pretty::dump_mir(tcx, "mir_map", &0, src, &mir);

            let mir = tcx.alloc_mir(mir);
            let def_id = tcx.map.local_def_id(src.item_id());
            assert!(tcx.mir_map.borrow_mut().insert(def_id, mir).is_none());
        });

        let body = self.tcx.map.body(body_id);
        self.visit_body(body);
    }
}

fn closure_self_ty<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                             closure_expr_id: ast::NodeId,
                             body_id: hir::BodyId)
                             -> Ty<'tcx> {
    let closure_ty = tcx.body_tables(body_id).node_id_to_type(closure_expr_id);

    // We're just hard-coding the idea that the signature will be
    // &self or &mut self and hence will have a bound region with
    // number 0, hokey.
    let region = ty::Region::ReFree(ty::FreeRegion {
        scope: tcx.region_maps.item_extent(body_id.node_id),
        bound_region: ty::BoundRegion::BrAnon(0),
    });
    let region = tcx.mk_region(region);

    match tcx.closure_kind(tcx.map.local_def_id(closure_expr_id)) {
        ty::ClosureKind::Fn =>
            tcx.mk_ref(region,
                       ty::TypeAndMut { ty: closure_ty,
                                        mutbl: hir::MutImmutable }),
        ty::ClosureKind::FnMut =>
            tcx.mk_ref(region,
                       ty::TypeAndMut { ty: closure_ty,
                                        mutbl: hir::MutMutable }),
        ty::ClosureKind::FnOnce =>
            closure_ty
    }
}
