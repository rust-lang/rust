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

extern crate syntax;

use build;
use rustc::dep_graph::DepNode;
use rustc::mir::repr::Mir;
use pretty;
use hair::cx::Cx;

use rustc::mir::mir_map::MirMap;
use rustc::infer;
use rustc::traits::ProjectionMode;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::util::nodemap::NodeMap;
use rustc::hir;
use rustc::hir::intravisit::{self, Visitor};
use syntax::ast;
use syntax::codemap::Span;

pub fn build_mir_for_crate<'tcx>(tcx: &TyCtxt<'tcx>) -> MirMap<'tcx> {
    let mut map = MirMap {
        map: NodeMap(),
    };
    {
        let mut dump = BuildMir {
            tcx: tcx,
            map: &mut map,
        };
        tcx.visit_all_items_in_krate(DepNode::MirMapConstruction, &mut dump);
    }
    map
}

///////////////////////////////////////////////////////////////////////////
// BuildMir -- walks a crate, looking for fn items and methods to build MIR from

struct BuildMir<'a, 'tcx: 'a> {
    tcx: &'a TyCtxt<'tcx>,
    map: &'a mut MirMap<'tcx>,
}

impl<'a, 'tcx> BuildMir<'a, 'tcx> {
    fn build<F>(&mut self, id: ast::NodeId, f: F)
        where F: for<'b> FnOnce(Cx<'b, 'tcx>) -> (Mir<'tcx>, build::ScopeAuxiliaryVec)
    {
        let param_env = ty::ParameterEnvironment::for_item(self.tcx, id);
        let infcx = infer::new_infer_ctxt(self.tcx,
                                          &self.tcx.tables,
                                          Some(param_env),
                                          ProjectionMode::AnyFinal);

        let (mir, scope_auxiliary) = f(Cx::new(&infcx));

        pretty::dump_mir(self.tcx, "mir_map", &0, id, &mir, Some(&scope_auxiliary));

        assert!(self.map.map.insert(id, mir).is_none())
    }
}

impl<'a, 'tcx> Visitor<'tcx> for BuildMir<'a, 'tcx> {
    fn visit_fn(&mut self,
                fk: intravisit::FnKind<'tcx>,
                decl: &'tcx hir::FnDecl,
                body: &'tcx hir::Block,
                span: Span,
                id: ast::NodeId) {
        // fetch the fully liberated fn signature (that is, all bound
        // types/lifetimes replaced)
        let fn_sig = match self.tcx.tables.borrow().liberated_fn_sigs.get(&id) {
            Some(f) => f.clone(),
            None => {
                span_bug!(span, "no liberated fn sig for {:?}", id);
            }
        };

        let implicit_argument = if let intravisit::FnKind::Closure(..) = fk {
            Some((closure_self_ty(&self.tcx, id, body.id), None))
        } else {
            None
        };

        let explicit_arguments =
            decl.inputs
                .iter()
                .enumerate()
                .map(|(index, arg)| {
                    (fn_sig.inputs[index], Some(&*arg.pat))
                });

        self.build(id, |cx| {
            let arguments = implicit_argument.into_iter().chain(explicit_arguments);
            build::construct_fn(cx, id, arguments, fn_sig.output, body)
        });

        intravisit::walk_fn(self, fk, decl, body, span);
    }
}

fn closure_self_ty<'a, 'tcx>(tcx: &TyCtxt<'tcx>,
                             closure_expr_id: ast::NodeId,
                             body_id: ast::NodeId)
                             -> Ty<'tcx> {
    let closure_ty = tcx.node_id_to_type(closure_expr_id);

    // We're just hard-coding the idea that the signature will be
    // &self or &mut self and hence will have a bound region with
    // number 0, hokey.
    let region = ty::Region::ReFree(ty::FreeRegion {
        scope: tcx.region_maps.item_extent(body_id),
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
