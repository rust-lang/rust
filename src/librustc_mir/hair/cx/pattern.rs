// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use hair::*;
use hair::cx::Cx;
use hair::cx::to_ref::ToRef;
use repr::*;
use rustc_data_structures::fnv::FnvHashMap;
use std::rc::Rc;
use rustc::middle::const_eval;
use rustc::middle::def;
use rustc::middle::pat_util::{pat_is_resolved_const, pat_is_binding};
use rustc::middle::subst::Substs;
use rustc::middle::ty::{self, Ty};
use rustc_front::hir;
use syntax::ast;
use syntax::ptr::P;

/// When there are multiple patterns in a single arm, each one has its
/// own node-ids for the bindings.  References to the variables always
/// use the node-ids from the first pattern in the arm, so we just
/// remap the ids for all subsequent bindings to the first one.
///
/// Example:
/// ```
/// match foo {
///    Test1(flavor /* def 1 */) |
///    Test2(flavor /* def 2 */) if flavor /* ref 1 */.is_tasty() => { ... }
///    _ => { ... }
/// }
/// ```
#[derive(Clone, Debug)]
pub struct PatNode<'tcx> {
    pat: &'tcx hir::Pat,
    binding_map: Option<Rc<FnvHashMap<ast::Name, ast::NodeId>>>,
}

impl<'tcx> PatNode<'tcx> {
    pub fn new(pat: &'tcx hir::Pat,
               binding_map: Option<Rc<FnvHashMap<ast::Name, ast::NodeId>>>)
               -> PatNode<'tcx> {
        PatNode {
            pat: pat,
            binding_map: binding_map,
        }
    }

    pub fn irrefutable(pat: &'tcx hir::Pat) -> PatNode<'tcx> {
        PatNode::new(pat, None)
    }

    fn pat_ref<'a>(&self, pat: &'tcx hir::Pat) -> PatternRef<'tcx> {
        PatNode::new(pat, self.binding_map.clone()).to_ref()
    }

    fn pat_refs<'a>(&self, pats: &'tcx Vec<P<hir::Pat>>) -> Vec<PatternRef<'tcx>> {
        pats.iter().map(|p| self.pat_ref(p)).collect()
    }

    fn opt_pat_ref<'a>(&self, pat: &'tcx Option<P<hir::Pat>>) -> Option<PatternRef<'tcx>> {
        pat.as_ref().map(|p| self.pat_ref(p))
    }

    fn slice_or_array_pattern<'a>(&self,
                                  cx: &mut Cx<'a, 'tcx>,
                                  ty: Ty<'tcx>,
                                  prefix: &'tcx Vec<P<hir::Pat>>,
                                  slice: &'tcx Option<P<hir::Pat>>,
                                  suffix: &'tcx Vec<P<hir::Pat>>)
                                  -> PatternKind<'tcx> {
        match ty.sty {
            ty::TySlice(..) =>
                // matching a slice or fixed-length array
                PatternKind::Slice {
                    prefix: self.pat_refs(prefix),
                    slice: self.opt_pat_ref(slice),
                    suffix: self.pat_refs(suffix),
                },

            ty::TyArray(_, len) => {
                // fixed-length array
                assert!(len >= prefix.len() + suffix.len());
                PatternKind::Array {
                    prefix: self.pat_refs(prefix),
                    slice: self.opt_pat_ref(slice),
                    suffix: self.pat_refs(suffix),
                }
            }

            _ => {
                cx.tcx.sess.span_bug(self.pat.span, "unexpanded macro or bad constant etc");
            }
        }
    }

    fn variant_or_leaf<'a>(&self,
                           cx: &mut Cx<'a, 'tcx>,
                           subpatterns: Vec<FieldPatternRef<'tcx>>)
                           -> PatternKind<'tcx> {
        let def = cx.tcx.def_map.borrow().get(&self.pat.id).unwrap().full_def();
        match def {
            def::DefVariant(enum_id, variant_id, _) => {
                let adt_def = cx.tcx.lookup_adt_def(enum_id);
                if adt_def.variants.len() > 1 {
                    PatternKind::Variant {
                        adt_def: adt_def,
                        variant_index: adt_def.variant_index_with_id(variant_id),
                        subpatterns: subpatterns,
                    }
                } else {
                    PatternKind::Leaf { subpatterns: subpatterns }
                }
            }

            // NB: resolving to DefStruct means the struct *constructor*,
            // not the struct as a type.
            def::DefStruct(..) | def::DefTy(..) => {
                PatternKind::Leaf { subpatterns: subpatterns }
            }

            _ => {
                cx.tcx.sess.span_bug(self.pat.span,
                                     &format!("inappropriate def for pattern: {:?}", def));
            }
        }
    }
}

impl<'tcx> Mirror<'tcx> for PatNode<'tcx> {
    type Output = Pattern<'tcx>;

    fn make_mirror<'a>(self, cx: &mut Cx<'a, 'tcx>) -> Pattern<'tcx> {
        let kind = match self.pat.node {
            hir::PatWild => PatternKind::Wild,

            hir::PatLit(ref value) => {
                let value = const_eval::eval_const_expr(cx.tcx, value);
                let value = Literal::Value { value: value };
                PatternKind::Constant { value: value }
            }

            hir::PatRange(ref lo, ref hi) => {
                let lo = const_eval::eval_const_expr(cx.tcx, lo);
                let lo = Literal::Value { value: lo };
                let hi = const_eval::eval_const_expr(cx.tcx, hi);
                let hi = Literal::Value { value: hi };
                PatternKind::Range { lo: lo, hi: hi }
            },

            hir::PatEnum(..) | hir::PatIdent(..) | hir::PatQPath(..)
                if pat_is_resolved_const(&cx.tcx.def_map.borrow(), self.pat) =>
            {
                let def = cx.tcx.def_map.borrow().get(&self.pat.id).unwrap().full_def();
                match def {
                    def::DefConst(def_id) | def::DefAssociatedConst(def_id) =>
                        match const_eval::lookup_const_by_id(cx.tcx, def_id, Some(self.pat.id)) {
                            Some(const_expr) => {
                                let opt_value =
                                    const_eval::eval_const_expr_partial(
                                        cx.tcx, const_expr,
                                        const_eval::EvalHint::ExprTypeChecked,
                                        None);
                                let literal = if let Ok(value) = opt_value {
                                    Literal::Value { value: value }
                                } else {
                                    let substs = cx.tcx.mk_substs(Substs::empty());
                                    Literal::Item { def_id: def_id, substs: substs }
                                };
                                PatternKind::Constant { value: literal }
                            }
                            None => {
                                cx.tcx.sess.span_bug(
                                    self.pat.span,
                                    &format!("cannot eval constant: {:?}", def_id))
                            }
                        },
                    _ =>
                        cx.tcx.sess.span_bug(
                            self.pat.span,
                            &format!("def not a constant: {:?}", def)),
                }
            }

            hir::PatRegion(ref subpattern, _) |
            hir::PatBox(ref subpattern) => {
                PatternKind::Deref { subpattern: self.pat_ref(subpattern) }
            }

            hir::PatVec(ref prefix, ref slice, ref suffix) => {
                let ty = cx.tcx.node_id_to_type(self.pat.id);
                match ty.sty {
                    ty::TyRef(_, mt) =>
                        PatternKind::Deref {
                            subpattern: Pattern {
                                ty: mt.ty,
                                span: self.pat.span,
                                kind: self.slice_or_array_pattern(cx, mt.ty, prefix,
                                                                  slice, suffix),
                            }.to_ref()
                        },

                    ty::TySlice(..) |
                    ty::TyArray(..) =>
                        self.slice_or_array_pattern(cx, ty, prefix, slice, suffix),

                    ref sty =>
                        cx.tcx.sess.span_bug(
                            self.pat.span,
                            &format!("unexpanded type for vector pattern: {:?}", sty)),
                }
            }

            hir::PatTup(ref subpatterns) => {
                let subpatterns =
                    subpatterns.iter()
                               .enumerate()
                               .map(|(i, subpattern)| FieldPatternRef {
                                   field: Field::new(i),
                                   pattern: self.pat_ref(subpattern),
                               })
                               .collect();

                PatternKind::Leaf { subpatterns: subpatterns }
            }

            hir::PatIdent(bm, ref ident, ref sub)
                if pat_is_binding(&cx.tcx.def_map.borrow(), self.pat) =>
            {
                let id = match self.binding_map {
                    None => self.pat.id,
                    Some(ref map) => map[&ident.node.name],
                };
                let var_ty = cx.tcx.node_id_to_type(self.pat.id);
                let region = match var_ty.sty {
                    ty::TyRef(&r, _) => Some(r),
                    _ => None,
                };
                let (mutability, mode) = match bm {
                    hir::BindByValue(hir::MutMutable) =>
                        (Mutability::Mut, BindingMode::ByValue),
                    hir::BindByValue(hir::MutImmutable) =>
                        (Mutability::Not, BindingMode::ByValue),
                    hir::BindByRef(hir::MutMutable) =>
                        (Mutability::Not, BindingMode::ByRef(region.unwrap(), BorrowKind::Mut)),
                    hir::BindByRef(hir::MutImmutable) =>
                        (Mutability::Not, BindingMode::ByRef(region.unwrap(), BorrowKind::Shared)),
                };
                PatternKind::Binding {
                    mutability: mutability,
                    mode: mode,
                    name: ident.node.name,
                    var: id,
                    ty: var_ty,
                    subpattern: self.opt_pat_ref(sub),
                }
            }

            hir::PatIdent(..) => {
                self.variant_or_leaf(cx, vec![])
            }

            hir::PatEnum(_, ref opt_subpatterns) => {
                let subpatterns =
                    opt_subpatterns.iter()
                                   .flat_map(|v| v.iter())
                                   .enumerate()
                                   .map(|(i, field)| FieldPatternRef {
                                       field: Field::new(i),
                                       pattern: self.pat_ref(field),
                                   })
                                   .collect();
                self.variant_or_leaf(cx, subpatterns)
            }

            hir::PatStruct(_, ref fields, _) => {
                let pat_ty = cx.tcx.node_id_to_type(self.pat.id);
                let adt_def = match pat_ty.sty {
                    ty::TyStruct(adt_def, _) | ty::TyEnum(adt_def, _) => adt_def,
                    _ => {
                        cx.tcx.sess.span_bug(
                            self.pat.span,
                            "struct pattern not applied to struct or enum");
                    }
                };

                let def = cx.tcx.def_map.borrow().get(&self.pat.id).unwrap().full_def();
                let variant_def = adt_def.variant_of_def(def);

                let subpatterns =
                    fields.iter()
                          .map(|field| {
                              let index = variant_def.index_of_field_named(field.node.name);
                              let index = index.unwrap_or_else(|| {
                                  cx.tcx.sess.span_bug(
                                      self.pat.span,
                                      &format!("no field with name {:?}", field.node.name));
                              });
                              FieldPatternRef {
                                  field: Field::new(index),
                                  pattern: self.pat_ref(&field.node.pat),
                              }
                          })
                          .collect();

                self.variant_or_leaf(cx, subpatterns)
            }

            hir::PatQPath(..) => {
                cx.tcx.sess.span_bug(self.pat.span, "unexpanded macro or bad constant etc");
            }
        };

        let ty = cx.tcx.node_id_to_type(self.pat.id);

        Pattern {
            span: self.pat.span,
            ty: ty,
            kind: kind,
        }
    }
}
