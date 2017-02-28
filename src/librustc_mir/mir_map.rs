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
use rustc::hir::def_id::DefId;
use rustc::dep_graph::DepNode;
use rustc::mir::Mir;
use rustc::mir::transform::MirSource;
use rustc::mir::visit::MutVisitor;
use pretty;
use hair::cx::Cx;

use rustc::traits::Reveal;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::maps::Providers;
use rustc::ty::subst::Substs;
use rustc::hir;
use syntax::abi::Abi;
use syntax::ast;
use syntax_pos::Span;

use std::cell::RefCell;
use std::mem;

pub fn build_mir_for_crate<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    tcx.dep_graph.with_task(DepNode::MirKrate, || {
        tcx.visit_all_bodies_in_krate(|body_owner_def_id, _body_id| {
            tcx.item_mir(body_owner_def_id);
        });
    });
}

pub fn provide(providers: &mut Providers) {
    providers.mir = build_mir;
}

fn build_mir<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId)
                       -> &'tcx RefCell<Mir<'tcx>> {
    let id = tcx.hir.as_local_node_id(def_id).unwrap();
    let unsupported = || {
        span_bug!(tcx.hir.span(id), "can't build MIR for {:?}", def_id);
    };

    // Figure out what primary body this item has.
    let body_id = match tcx.hir.get(id) {
        hir::map::NodeItem(item) => {
            match item.node {
                hir::ItemConst(_, body) |
                hir::ItemStatic(_, _, body) |
                hir::ItemFn(.., body) => body,
                _ => unsupported()
            }
        }
        hir::map::NodeTraitItem(item) => {
            match item.node {
                hir::TraitItemKind::Const(_, Some(body)) |
                hir::TraitItemKind::Method(_,
                    hir::TraitMethod::Provided(body)) => body,
                _ => unsupported()
            }
        }
        hir::map::NodeImplItem(item) => {
            match item.node {
                hir::ImplItemKind::Const(_, body) |
                hir::ImplItemKind::Method(_, body) => body,
                _ => unsupported()
            }
        }
        hir::map::NodeExpr(expr) => {
            // FIXME(eddyb) Closures should have separate
            // function definition IDs and expression IDs.
            // Type-checking should not let closures get
            // this far in a constant position.
            // Assume that everything other than closures
            // is a constant "initializer" expression.
            match expr.node {
                hir::ExprClosure(_, _, body, _) => body,
                _ => hir::BodyId { node_id: expr.id }
            }
        }
        _ => unsupported()
    };

    let src = MirSource::from_node(tcx, id);
    tcx.infer_ctxt(body_id, Reveal::UserFacing).enter(|infcx| {
        let cx = Cx::new(&infcx, src);
        let mut mir = if cx.tables().tainted_by_errors {
            build::construct_error(cx, body_id)
        } else if let MirSource::Fn(id) = src {
            // fetch the fully liberated fn signature (that is, all bound
            // types/lifetimes replaced)
            let fn_sig = cx.tables().liberated_fn_sigs[&id].clone();

            let ty = tcx.item_type(tcx.hir.local_def_id(id));
            let mut abi = fn_sig.abi;
            let implicit_argument = if let ty::TyClosure(..) = ty.sty {
                // HACK(eddyb) Avoid having RustCall on closures,
                // as it adds unnecessary (and wrong) auto-tupling.
                abi = Abi::Rust;
                Some((closure_self_ty(tcx, id, body_id), None))
            } else {
                None
            };

            let body = tcx.hir.body(body_id);
            let explicit_arguments =
                body.arguments
                    .iter()
                    .enumerate()
                    .map(|(index, arg)| {
                        (fn_sig.inputs()[index], Some(&*arg.pat))
                    });

            let arguments = implicit_argument.into_iter().chain(explicit_arguments);
            build::construct_fn(cx, id, arguments, abi, fn_sig.output(), body)
        } else {
            build::construct_const(cx, body_id)
        };

        // Convert the Mir to global types.
        let mut globalizer = GlobalizeMir {
            tcx: tcx,
            span: mir.span
        };
        globalizer.visit_mir(&mut mir);
        let mir = unsafe {
            mem::transmute::<Mir, Mir<'tcx>>(mir)
        };

        pretty::dump_mir(tcx, "mir_map", &0, src, &mir);

        tcx.alloc_mir(mir)
    })
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

    match tcx.closure_kind(tcx.hir.local_def_id(closure_expr_id)) {
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
