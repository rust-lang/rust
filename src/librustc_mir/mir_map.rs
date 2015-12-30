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
extern crate rustc;
extern crate rustc_front;

use build;
use graphviz;
use pretty;
use transform::*;
use rustc::mir::repr::Mir;
use hair::cx::Cx;
use std::fs::File;

use self::rustc::middle::infer;
use self::rustc::middle::region::CodeExtentData;
use self::rustc::middle::ty::{self, Ty};
use self::rustc::util::common::ErrorReported;
use self::rustc::util::nodemap::NodeMap;
use self::rustc_front::hir;
use self::rustc_front::intravisit::{self, Visitor};
use self::syntax::ast;
use self::syntax::attr::AttrMetaMethods;
use self::syntax::codemap::Span;

pub type MirMap<'tcx> = NodeMap<Mir<'tcx>>;

pub fn build_mir_for_crate<'tcx>(tcx: &ty::ctxt<'tcx>) -> MirMap<'tcx> {
    let mut map = NodeMap();
    {
        let mut dump = OuterDump {
            tcx: tcx,
            map: &mut map,
        };
        tcx.map.krate().visit_all_items(&mut dump);
    }
    map
}

///////////////////////////////////////////////////////////////////////////
// OuterDump -- walks a crate, looking for fn items and methods to build MIR from

struct OuterDump<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
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
    tcx: &'a ty::ctxt<'tcx>,
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
        let (prefix, implicit_arg_tys) = match fk {
            intravisit::FnKind::Closure =>
                (format!("{}-", id), vec![closure_self_ty(&self.tcx, id, body.id)]),
            _ =>
                (format!(""), vec![]),
        };

        let param_env = ty::ParameterEnvironment::for_item(self.tcx, id);

        let infcx = infer::new_infer_ctxt(self.tcx, &self.tcx.tables, Some(param_env), true);

        match build_mir(Cx::new(&infcx), implicit_arg_tys, id, span, decl, body) {
            Ok(mut mir) => {
                simplify_cfg::SimplifyCfg::new().run_on_mir(&mut mir);

                let meta_item_list = self.attr
                                         .iter()
                                         .flat_map(|a| a.meta_item_list())
                                         .flat_map(|l| l.iter());
                for item in meta_item_list {
                    if item.check_name("graphviz") || item.check_name("pretty") {
                        match item.value_str() {
                            Some(s) => {
                                let filename = format!("{}{}", prefix, s);
                                let result = File::create(&filename).and_then(|ref mut output| {
                                    if item.check_name("graphviz") {
                                        graphviz::write_mir_graphviz(&mir, output)
                                    } else {
                                        pretty::write_mir_pretty(&mir, output)
                                    }
                                });

                                if let Err(e) = result {
                                    self.tcx.sess.span_fatal(
                                        item.span,
                                        &format!("Error writing MIR {} results to `{}`: {}",
                                                 item.name(), filename, e));
                                }
                            }
                            None => {
                                self.tcx.sess.span_err(
                                    item.span,
                                    &format!("{} attribute requires a path", item.name()));
                            }
                        }
                    }
                }

                let previous = self.map.insert(id, mir);
                assert!(previous.is_none());
            }
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
            cx.tcx().sess.span_bug(span,
                                   &format!("no liberated fn sig for {:?}", fn_id));
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

    let parameter_scope =
        cx.tcx().region_maps.lookup_code_extent(
            CodeExtentData::ParameterScope { fn_id: fn_id, body_id: body.id });
    Ok(build::construct(cx,
                        span,
                        implicit_arg_tys,
                        arguments,
                        parameter_scope,
                        fn_sig.output,
                        body))
}

fn closure_self_ty<'a, 'tcx>(tcx: &ty::ctxt<'tcx>,
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
        ty::ClosureKind::FnClosureKind =>
            tcx.mk_ref(region,
                       ty::TypeAndMut { ty: closure_ty,
                                        mutbl: hir::MutImmutable }),
        ty::ClosureKind::FnMutClosureKind =>
            tcx.mk_ref(region,
                       ty::TypeAndMut { ty: closure_ty,
                                        mutbl: hir::MutMutable }),
        ty::ClosureKind::FnOnceClosureKind =>
            closure_ty
    }
}
