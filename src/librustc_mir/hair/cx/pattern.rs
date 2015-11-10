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
use repr::*;
use rustc_data_structures::fnv::FnvHashMap;
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
struct PatCx<'patcx, 'cx: 'patcx, 'tcx: 'cx> {
    cx: &'patcx mut Cx<'cx, 'tcx>,
    binding_map: Option<&'patcx FnvHashMap<ast::Name, ast::NodeId>>,
}

impl<'cx, 'tcx> Cx<'cx, 'tcx> {
    pub fn irrefutable_pat(&mut self, pat: &'tcx hir::Pat) -> Pattern<'tcx> {
        PatCx::new(self, None).to_pat(pat)
    }

    pub fn refutable_pat(&mut self,
                         binding_map: Option<&FnvHashMap<ast::Name, ast::NodeId>>,
                         pat: &'tcx hir::Pat)
                         -> Pattern<'tcx> {
        PatCx::new(self, binding_map).to_pat(pat)
    }
}

impl<'patcx, 'cx, 'tcx> PatCx<'patcx, 'cx, 'tcx> {
    fn new(cx: &'patcx mut Cx<'cx, 'tcx>,
               binding_map: Option<&'patcx FnvHashMap<ast::Name, ast::NodeId>>)
               -> PatCx<'patcx, 'cx, 'tcx> {
        PatCx {
            cx: cx,
            binding_map: binding_map,
        }
    }

    fn to_pat(&mut self, pat: &'tcx hir::Pat) -> Pattern<'tcx> {
        let kind = match pat.node {
            hir::PatWild => PatternKind::Wild,

            hir::PatLit(ref value) => {
                let value = const_eval::eval_const_expr(self.cx.tcx, value);
                let value = Literal::Value { value: value };
                PatternKind::Constant { value: value }
            }

            hir::PatRange(ref lo, ref hi) => {
                let lo = const_eval::eval_const_expr(self.cx.tcx, lo);
                let lo = Literal::Value { value: lo };
                let hi = const_eval::eval_const_expr(self.cx.tcx, hi);
                let hi = Literal::Value { value: hi };
                PatternKind::Range { lo: lo, hi: hi }
            },

            hir::PatEnum(..) | hir::PatIdent(..) | hir::PatQPath(..)
                if pat_is_resolved_const(&self.cx.tcx.def_map.borrow(), pat) =>
            {
                let def = self.cx.tcx.def_map.borrow().get(&pat.id).unwrap().full_def();
                match def {
                    def::DefConst(def_id) | def::DefAssociatedConst(def_id) =>
                        match const_eval::lookup_const_by_id(self.cx.tcx, def_id, Some(pat.id)) {
                            Some(const_expr) => {
                                let opt_value =
                                    const_eval::eval_const_expr_partial(
                                        self.cx.tcx, const_expr,
                                        const_eval::EvalHint::ExprTypeChecked,
                                        None);
                                let literal = if let Ok(value) = opt_value {
                                    Literal::Value { value: value }
                                } else {
                                    let substs = self.cx.tcx.mk_substs(Substs::empty());
                                    Literal::Item { def_id: def_id, substs: substs }
                                };
                                PatternKind::Constant { value: literal }
                            }
                            None => {
                                self.cx.tcx.sess.span_bug(
                                    pat.span,
                                    &format!("cannot eval constant: {:?}", def_id))
                            }
                        },
                    _ =>
                        self.cx.tcx.sess.span_bug(
                            pat.span,
                            &format!("def not a constant: {:?}", def)),
                }
            }

            hir::PatRegion(ref subpattern, _) |
            hir::PatBox(ref subpattern) => {
                PatternKind::Deref { subpattern: self.to_pat(subpattern) }
            }

            hir::PatVec(ref prefix, ref slice, ref suffix) => {
                let ty = self.cx.tcx.node_id_to_type(pat.id);
                match ty.sty {
                    ty::TyRef(_, mt) =>
                        PatternKind::Deref {
                            subpattern: Pattern {
                                ty: mt.ty,
                                span: pat.span,
                                kind: Box::new(self.slice_or_array_pattern(pat, mt.ty, prefix,
                                                                           slice, suffix)),
                            },
                        },

                    ty::TySlice(..) |
                    ty::TyArray(..) =>
                        self.slice_or_array_pattern(pat, ty, prefix, slice, suffix),

                    ref sty =>
                        self.cx.tcx.sess.span_bug(
                            pat.span,
                            &format!("unexpanded type for vector pattern: {:?}", sty)),
                }
            }

            hir::PatTup(ref subpatterns) => {
                let subpatterns =
                    subpatterns.iter()
                               .enumerate()
                               .map(|(i, subpattern)| FieldPattern {
                                   field: Field::new(i),
                                   pattern: self.to_pat(subpattern),
                               })
                               .collect();

                PatternKind::Leaf { subpatterns: subpatterns }
            }

            hir::PatIdent(bm, ref ident, ref sub)
                if pat_is_binding(&self.cx.tcx.def_map.borrow(), pat) =>
            {
                let id = match self.binding_map {
                    None => pat.id,
                    Some(ref map) => map[&ident.node.name],
                };
                let var_ty = self.cx.tcx.node_id_to_type(pat.id);
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
                    subpattern: self.to_opt_pat(sub),
                }
            }

            hir::PatIdent(..) => {
                self.variant_or_leaf(pat, vec![])
            }

            hir::PatEnum(_, ref opt_subpatterns) => {
                let subpatterns =
                    opt_subpatterns.iter()
                                   .flat_map(|v| v.iter())
                                   .enumerate()
                                   .map(|(i, field)| FieldPattern {
                                       field: Field::new(i),
                                       pattern: self.to_pat(field),
                                   })
                                   .collect();
                self.variant_or_leaf(pat, subpatterns)
            }

            hir::PatStruct(_, ref fields, _) => {
                let pat_ty = self.cx.tcx.node_id_to_type(pat.id);
                let adt_def = match pat_ty.sty {
                    ty::TyStruct(adt_def, _) | ty::TyEnum(adt_def, _) => adt_def,
                    _ => {
                        self.cx.tcx.sess.span_bug(
                            pat.span,
                            "struct pattern not applied to struct or enum");
                    }
                };

                let def = self.cx.tcx.def_map.borrow().get(&pat.id).unwrap().full_def();
                let variant_def = adt_def.variant_of_def(def);

                let subpatterns =
                    fields.iter()
                          .map(|field| {
                              let index = variant_def.index_of_field_named(field.node.name);
                              let index = index.unwrap_or_else(|| {
                                  self.cx.tcx.sess.span_bug(
                                      pat.span,
                                      &format!("no field with name {:?}", field.node.name));
                              });
                              FieldPattern {
                                  field: Field::new(index),
                                  pattern: self.to_pat(&field.node.pat),
                              }
                          })
                          .collect();

                self.variant_or_leaf(pat, subpatterns)
            }

            hir::PatQPath(..) => {
                self.cx.tcx.sess.span_bug(pat.span, "unexpanded macro or bad constant etc");
            }
        };

        let ty = self.cx.tcx.node_id_to_type(pat.id);

        Pattern {
            span: pat.span,
            ty: ty,
            kind: Box::new(kind),
        }
    }

    fn to_pats(&mut self, pats: &'tcx Vec<P<hir::Pat>>) -> Vec<Pattern<'tcx>> {
        pats.iter().map(|p| self.to_pat(p)).collect()
    }

    fn to_opt_pat(&mut self, pat: &'tcx Option<P<hir::Pat>>) -> Option<Pattern<'tcx>> {
        pat.as_ref().map(|p| self.to_pat(p))
    }

    fn slice_or_array_pattern(&mut self,
                              pat: &'tcx hir::Pat,
                              ty: Ty<'tcx>,
                              prefix: &'tcx Vec<P<hir::Pat>>,
                              slice: &'tcx Option<P<hir::Pat>>,
                              suffix: &'tcx Vec<P<hir::Pat>>)
                              -> PatternKind<'tcx> {
        match ty.sty {
            ty::TySlice(..) => {
                // matching a slice or fixed-length array
                PatternKind::Slice {
                    prefix: self.to_pats(prefix),
                    slice: self.to_opt_pat(slice),
                    suffix: self.to_pats(suffix),
                }
            }

            ty::TyArray(_, len) => {
                // fixed-length array
                assert!(len >= prefix.len() + suffix.len());
                PatternKind::Array {
                    prefix: self.to_pats(prefix),
                    slice: self.to_opt_pat(slice),
                    suffix: self.to_pats(suffix),
                }
            }

            _ => {
                self.cx.tcx.sess.span_bug(pat.span, "unexpanded macro or bad constant etc");
            }
        }
    }

    fn variant_or_leaf(&mut self,
                       pat: &'tcx hir::Pat,
                       subpatterns: Vec<FieldPattern<'tcx>>)
                       -> PatternKind<'tcx> {
        let def = self.cx.tcx.def_map.borrow().get(&pat.id).unwrap().full_def();
        match def {
            def::DefVariant(enum_id, variant_id, _) => {
                let adt_def = self.cx.tcx.lookup_adt_def(enum_id);
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
                self.cx.tcx.sess.span_bug(pat.span,
                                          &format!("inappropriate def for pattern: {:?}", def));
            }
        }
    }
}
