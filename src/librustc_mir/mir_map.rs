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
use rustc::hir::def_id::DefId;
use rustc::mir::Mir;
use rustc::mir::transform::MirSource;
use rustc::mir::visit::MutVisitor;
use pretty;
use hair::cx::Cx;

use rustc::infer::InferCtxtBuilder;
use rustc::traits::Reveal;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::subst::Substs;
use rustc::hir;
use rustc::hir::intravisit::{self, FnKind, Visitor, NestedVisitorMap};
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

/// Helper type of a temporary returned by BuildMir::cx(...).
/// Necessary because we can't write the following bound:
/// F: for<'b, 'tcx> where 'gcx: 'tcx FnOnce(Cx<'b, 'gcx, 'tcx>).
struct CxBuilder<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    src: MirSource,
    def_id: DefId,
    infcx: InferCtxtBuilder<'a, 'gcx, 'tcx>
}

impl<'a, 'gcx, 'tcx> BuildMir<'a, 'gcx> {
    fn cx<'b>(&'b mut self, src: MirSource) -> CxBuilder<'b, 'gcx, 'tcx> {
        let param_env = ty::ParameterEnvironment::for_item(self.tcx, src.item_id());
        let def_id = self.tcx.map.local_def_id(src.item_id());
        CxBuilder {
            src: src,
            infcx: self.tcx.infer_ctxt(None, Some(param_env), Reveal::NotSpecializable),
            def_id: def_id
        }
    }
}

impl<'a, 'gcx, 'tcx> CxBuilder<'a, 'gcx, 'tcx> {
    fn build<F>(&'tcx mut self, f: F)
        where F: for<'b> FnOnce(Cx<'b, 'gcx, 'tcx>) -> Mir<'tcx>
    {
        let (src, def_id) = (self.src, self.def_id);
        self.infcx.enter(|infcx| {
            let mut mir = f(Cx::new(&infcx, src));

            // Convert the Mir to global types.
            let tcx = infcx.tcx.global_tcx();
            let mut globalizer = GlobalizeMir {
                tcx: tcx,
                span: mir.span
            };
            globalizer.visit_mir(&mut mir);
            let mir = unsafe {
                mem::transmute::<Mir, Mir<'gcx>>(mir)
            };

            pretty::dump_mir(tcx, "mir_map", &0, src, &mir);

            let mir = tcx.alloc_mir(mir);
            assert!(tcx.mir_map.borrow_mut().insert(def_id, mir).is_none());
        });
    }
}

impl<'a, 'gcx> BuildMir<'a, 'gcx> {
    fn build_const_integer(&mut self, expr: &'gcx hir::Expr) {
        // FIXME(eddyb) Closures should have separate
        // function definition IDs and expression IDs.
        // Type-checking should not let closures get
        // this far in an integer constant position.
        if let hir::ExprClosure(..) = expr.node {
            return;
        }
        self.cx(MirSource::Const(expr.id)).build(|cx| {
            build::construct_const(cx, expr.id, expr)
        });
    }
}

impl<'a, 'tcx> Visitor<'tcx> for BuildMir<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::OnlyBodies(&self.tcx.map)
    }

    // Const and static items.
    fn visit_item(&mut self, item: &'tcx hir::Item) {
        match item.node {
            hir::ItemConst(_, ref expr) => {
                self.cx(MirSource::Const(item.id)).build(|cx| {
                    build::construct_const(cx, item.id, expr)
                });
            }
            hir::ItemStatic(_, m, ref expr) => {
                self.cx(MirSource::Static(item.id, m)).build(|cx| {
                    build::construct_const(cx, item.id, expr)
                });
            }
            _ => {}
        }
        intravisit::walk_item(self, item);
    }

    // Trait associated const defaults.
    fn visit_trait_item(&mut self, item: &'tcx hir::TraitItem) {
        if let hir::ConstTraitItem(_, Some(ref expr)) = item.node {
            self.cx(MirSource::Const(item.id)).build(|cx| {
                build::construct_const(cx, item.id, expr)
            });
        }
        intravisit::walk_trait_item(self, item);
    }

    // Impl associated const.
    fn visit_impl_item(&mut self, item: &'tcx hir::ImplItem) {
        if let hir::ImplItemKind::Const(_, ref expr) = item.node {
            self.cx(MirSource::Const(item.id)).build(|cx| {
                build::construct_const(cx, item.id, expr)
            });
        }
        intravisit::walk_impl_item(self, item);
    }

    // Repeat counts, i.e. [expr; constant].
    fn visit_expr(&mut self, expr: &'tcx hir::Expr) {
        if let hir::ExprRepeat(_, ref count) = expr.node {
            self.build_const_integer(count);
        }
        intravisit::walk_expr(self, expr);
    }

    // Array lengths, i.e. [T; constant].
    fn visit_ty(&mut self, ty: &'tcx hir::Ty) {
        if let hir::TyArray(_, ref length) = ty.node {
            self.build_const_integer(length);
        }
        intravisit::walk_ty(self, ty);
    }

    // Enum variant discriminant values.
    fn visit_variant(&mut self, v: &'tcx hir::Variant,
                     g: &'tcx hir::Generics, item_id: ast::NodeId) {
        if let Some(ref expr) = v.node.disr_expr {
            self.build_const_integer(expr);
        }
        intravisit::walk_variant(self, v, g, item_id);
    }

    fn visit_fn(&mut self,
                fk: FnKind<'tcx>,
                decl: &'tcx hir::FnDecl,
                body_id: hir::ExprId,
                span: Span,
                id: ast::NodeId) {
        // fetch the fully liberated fn signature (that is, all bound
        // types/lifetimes replaced)
        let fn_sig = match self.tcx.tables().liberated_fn_sigs.get(&id) {
            Some(f) => f.clone(),
            None => {
                span_bug!(span, "no liberated fn sig for {:?}", id);
            }
        };

        let (abi, implicit_argument) = if let FnKind::Closure(..) = fk {
            (Abi::Rust, Some((closure_self_ty(self.tcx, id, body_id.node_id()), None)))
        } else {
            let def_id = self.tcx.map.local_def_id(id);
            (self.tcx.item_type(def_id).fn_abi(), None)
        };

        let explicit_arguments =
            decl.inputs
                .iter()
                .enumerate()
                .map(|(index, arg)| {
                    (fn_sig.inputs()[index], Some(&*arg.pat))
                });

        let body = self.tcx.map.expr(body_id);

        let arguments = implicit_argument.into_iter().chain(explicit_arguments);
        self.cx(MirSource::Fn(id)).build(|cx| {
            build::construct_fn(cx, id, arguments, abi, fn_sig.output(), body)
        });

        intravisit::walk_fn(self, fk, decl, body_id, span, id);
    }
}

fn closure_self_ty<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                             closure_expr_id: ast::NodeId,
                             body_id: ast::NodeId)
                             -> Ty<'tcx> {
    let closure_ty = tcx.tables().node_id_to_type(closure_expr_id);

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
