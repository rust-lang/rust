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
use rustc::util::common::ErrorReported;
use rustc::util::nodemap::NodeMap;
use rustc::hir;
use rustc::hir::intravisit::{self, Visitor};
use syntax::abi::Abi;
use syntax::ast;
use syntax::attr::AttrMetaMethods;
use syntax::codemap::Span;

pub fn build_mir_for_crate<'tcx>(tcx: &TyCtxt<'tcx>) -> MirMap<'tcx> {
    let mut map = MirMap {
        map: NodeMap(),
    };
    {
        let mut dump = OuterDump {
            tcx: tcx,
            map: &mut map,
        };
        tcx.visit_all_items_in_krate(DepNode::MirMapConstruction, &mut dump);
    }
    map
}

///////////////////////////////////////////////////////////////////////////
// OuterDump -- walks a crate, looking for fn items and methods to build MIR from

struct OuterDump<'a, 'tcx: 'a> {
    tcx: &'a TyCtxt<'tcx>,
    map: &'a mut MirMap<'tcx>,
}

impl<'a, 'tcx> OuterDump<'a, 'tcx> {
    fn visit_mir<OP>(&mut self, attributes: &'a [ast::Attribute], mut walk_op: OP)
        where OP: for<'m> FnMut(&mut InnerDump<'a, 'm, 'tcx>)
    {
        let mut closure_dump = InnerDump {
            tcx: self.tcx,
            attr: None,
            map: &mut *self.map,
        };
        for attr in attributes {
            if attr.check_name("rustc_mir") {
                closure_dump.attr = Some(attr);
            }
        }
        walk_op(&mut closure_dump);
    }
}


impl<'a, 'tcx> Visitor<'tcx> for OuterDump<'a, 'tcx> {
    fn visit_item(&mut self, item: &'tcx hir::Item) {
        self.visit_mir(&item.attrs, |c| intravisit::walk_item(c, item));
        intravisit::walk_item(self, item);
    }

    fn visit_trait_item(&mut self, trait_item: &'tcx hir::TraitItem) {
        match trait_item.node {
            hir::MethodTraitItem(_, Some(_)) => {
                self.visit_mir(&trait_item.attrs, |c| intravisit::walk_trait_item(c, trait_item));
            }
            hir::MethodTraitItem(_, None) |
            hir::ConstTraitItem(..) |
            hir::TypeTraitItem(..) => {}
        }
        intravisit::walk_trait_item(self, trait_item);
    }

    fn visit_impl_item(&mut self, impl_item: &'tcx hir::ImplItem) {
        match impl_item.node {
            hir::ImplItemKind::Method(..) => {
                self.visit_mir(&impl_item.attrs, |c| intravisit::walk_impl_item(c, impl_item));
            }
            hir::ImplItemKind::Const(..) | hir::ImplItemKind::Type(..) => {}
        }
        intravisit::walk_impl_item(self, impl_item);
    }
}

///////////////////////////////////////////////////////////////////////////
// InnerDump -- dumps MIR for a single fn and its contained closures

struct InnerDump<'a, 'm, 'tcx: 'a + 'm> {
    tcx: &'a TyCtxt<'tcx>,
    map: &'m mut MirMap<'tcx>,
    attr: Option<&'a ast::Attribute>,
}

impl<'a, 'm, 'tcx> Visitor<'tcx> for InnerDump<'a,'m,'tcx> {
    fn visit_trait_item(&mut self, _: &'tcx hir::TraitItem) {
        // ignore methods; the outer dump will call us for them independently
    }

    fn visit_impl_item(&mut self, _: &'tcx hir::ImplItem) {
        // ignore methods; the outer dump will call us for them independently
    }

    fn visit_fn(&mut self,
                fk: intravisit::FnKind<'tcx>,
                decl: &'tcx hir::FnDecl,
                body: &'tcx hir::Block,
                span: Span,
                id: ast::NodeId) {
        let implicit_arg_tys = if let intravisit::FnKind::Closure(..) = fk {
            vec![closure_self_ty(&self.tcx, id, body.id)]
        } else {
            vec![]
        };

        let param_env = ty::ParameterEnvironment::for_item(self.tcx, id);
        let infcx = infer::new_infer_ctxt(self.tcx,
                                          &self.tcx.tables,
                                          Some(param_env),
                                          ProjectionMode::AnyFinal);

        match build_mir(Cx::new(&infcx), implicit_arg_tys, id, span, decl, body) {
            Ok(mir) => assert!(self.map.map.insert(id, mir).is_none()),
            Err(ErrorReported) => {}
        }

        intravisit::walk_fn(self, fk, decl, body, span);
    }
}

fn build_mir<'a,'tcx:'a>(cx: Cx<'a,'tcx>,
                         implicit_arg_tys: Vec<Ty<'tcx>>,
                         fn_id: ast::NodeId,
                         span: Span,
                         decl: &'tcx hir::FnDecl,
                         body: &'tcx hir::Block)
                         -> Result<Mir<'tcx>, ErrorReported> {
    // fetch the fully liberated fn signature (that is, all bound
    // types/lifetimes replaced)
    let fn_sig = match cx.tcx().tables.borrow().liberated_fn_sigs.get(&fn_id) {
        Some(f) => f.clone(),
        None => {
            span_bug!(span, "no liberated fn sig for {:?}", fn_id);
        }
    };

    let arguments =
        decl.inputs
            .iter()
            .enumerate()
            .map(|(index, arg)| {
                (fn_sig.inputs[index], &*arg.pat)
            })
            .collect();

    let (mut mir, scope_auxiliary) =
        build::construct(cx,
                         span,
                         fn_id,
                         body.id,
                         implicit_arg_tys,
                         arguments,
                         fn_sig.output,
                         body);

    match cx.tcx().node_id_to_type(fn_id).sty {
        ty::TyFnDef(_, _, f) if f.abi == Abi::RustCall => {
            // RustCall pseudo-ABI untuples the last argument.
            if let Some(arg_decl) = mir.arg_decls.last_mut() {
                arg_decl.spread = true;
            }
        }
        _ => {}
    }

    pretty::dump_mir(cx.tcx(),
                     "mir_map",
                     &0,
                     fn_id,
                     &mir,
                     Some(&scope_auxiliary));

    Ok(mir)
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
