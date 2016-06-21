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
use rustc_data_structures::indexed_vec::Idx;
use rustc_const_eval as const_eval;
use rustc::hir::def::Def;
use rustc::hir::pat_util::{EnumerateAndAdjustIterator, pat_is_resolved_const};
use rustc::ty::{self, Ty};
use rustc::mir::repr::*;
use rustc::hir::{self, PatKind};
use syntax::ptr::P;
use syntax_pos::Span;

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
struct PatCx<'patcx, 'cx: 'patcx, 'gcx: 'cx+'tcx, 'tcx: 'cx> {
    cx: &'patcx mut Cx<'cx, 'gcx, 'tcx>,
}

impl<'cx, 'gcx, 'tcx> Cx<'cx, 'gcx, 'tcx> {
    pub fn irrefutable_pat(&mut self, pat: &hir::Pat) -> Pattern<'tcx> {
        PatCx::new(self).to_pattern(pat)
    }

    pub fn refutable_pat(&mut self,
                         pat: &hir::Pat)
                         -> Pattern<'tcx> {
        PatCx::new(self).to_pattern(pat)
    }
}

impl<'patcx, 'cx, 'gcx, 'tcx> PatCx<'patcx, 'cx, 'gcx, 'tcx> {
    fn new(cx: &'patcx mut Cx<'cx, 'gcx, 'tcx>)
               -> PatCx<'patcx, 'cx, 'gcx, 'tcx> {
        PatCx {
            cx: cx,
        }
    }

    fn to_pattern(&mut self, pat: &hir::Pat) -> Pattern<'tcx> {
        let mut ty = self.cx.tcx.node_id_to_type(pat.id);

        let kind = match pat.node {
            PatKind::Wild => PatternKind::Wild,

            PatKind::Lit(ref value) => {
                let value = const_eval::eval_const_expr(self.cx.tcx.global_tcx(), value);
                PatternKind::Constant { value: value }
            }

            PatKind::Range(ref lo, ref hi) => {
                let lo = const_eval::eval_const_expr(self.cx.tcx.global_tcx(), lo);
                let lo = Literal::Value { value: lo };
                let hi = const_eval::eval_const_expr(self.cx.tcx.global_tcx(), hi);
                let hi = Literal::Value { value: hi };
                PatternKind::Range { lo: lo, hi: hi }
            },

            PatKind::Path(..) | PatKind::QPath(..)
                if pat_is_resolved_const(&self.cx.tcx.def_map.borrow(), pat) =>
            {
                match self.cx.tcx.expect_def(pat.id) {
                    Def::Const(def_id) | Def::AssociatedConst(def_id) => {
                        let tcx = self.cx.tcx.global_tcx();
                        let substs = Some(self.cx.tcx.node_id_item_substs(pat.id).substs);
                        match const_eval::lookup_const_by_id(tcx, def_id, substs) {
                            Some((const_expr, _const_ty)) => {
                                match const_eval::const_expr_to_pat(tcx,
                                                                    const_expr,
                                                                    pat.id,
                                                                    pat.span) {
                                    Ok(pat) =>
                                        return self.to_pattern(&pat),
                                    Err(_) =>
                                        span_bug!(
                                            pat.span, "illegal constant"),
                                }
                            }
                            None => {
                                span_bug!(
                                    pat.span,
                                    "cannot eval constant: {:?}",
                                    def_id)
                            }
                        }
                    }
                    def =>
                        span_bug!(
                            pat.span,
                            "def not a constant: {:?}",
                            def),
                }
            }

            PatKind::Ref(ref subpattern, _) |
            PatKind::Box(ref subpattern) => {
                PatternKind::Deref { subpattern: self.to_pattern(subpattern) }
            }

            PatKind::Vec(ref prefix, ref slice, ref suffix) => {
                let ty = self.cx.tcx.node_id_to_type(pat.id);
                match ty.sty {
                    ty::TyRef(_, mt) =>
                        PatternKind::Deref {
                            subpattern: Pattern {
                                ty: mt.ty,
                                span: pat.span,
                                kind: Box::new(self.slice_or_array_pattern(pat.span, mt.ty, prefix,
                                                                           slice, suffix)),
                            },
                        },

                    ty::TySlice(..) |
                    ty::TyArray(..) =>
                        self.slice_or_array_pattern(pat.span, ty, prefix, slice, suffix),

                    ref sty =>
                        span_bug!(
                            pat.span,
                            "unexpanded type for vector pattern: {:?}",
                            sty),
                }
            }

            PatKind::Tuple(ref subpatterns, ddpos) => {
                match self.cx.tcx.node_id_to_type(pat.id).sty {
                    ty::TyTuple(ref tys) => {
                        let subpatterns =
                            subpatterns.iter()
                                       .enumerate_and_adjust(tys.len(), ddpos)
                                       .map(|(i, subpattern)| FieldPattern {
                                            field: Field::new(i),
                                            pattern: self.to_pattern(subpattern),
                                       })
                                       .collect();

                        PatternKind::Leaf { subpatterns: subpatterns }
                    }

                    ref sty => span_bug!(pat.span, "unexpected type for tuple pattern: {:?}", sty),
                }
            }

            PatKind::Binding(bm, ref ident, ref sub) => {
                let id = self.cx.tcx.expect_def(pat.id).var_id();
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

                // A ref x pattern is the same node used for x, and as such it has
                // x's type, which is &T, where we want T (the type being matched).
                if let hir::BindByRef(_) = bm {
                    if let ty::TyRef(_, mt) = ty.sty {
                        ty = mt.ty;
                    } else {
                        bug!("`ref {}` has wrong type {}", ident.node, ty);
                    }
                }

                PatternKind::Binding {
                    mutability: mutability,
                    mode: mode,
                    name: ident.node,
                    var: id,
                    ty: var_ty,
                    subpattern: self.to_opt_pattern(sub),
                }
            }

            PatKind::Path(..) => {
                self.variant_or_leaf(pat, vec![])
            }

            PatKind::TupleStruct(_, ref subpatterns, ddpos) => {
                let pat_ty = self.cx.tcx.node_id_to_type(pat.id);
                let adt_def = match pat_ty.sty {
                    ty::TyStruct(adt_def, _) | ty::TyEnum(adt_def, _) => adt_def,
                    _ => span_bug!(pat.span, "tuple struct pattern not applied to struct or enum"),
                };
                let variant_def = adt_def.variant_of_def(self.cx.tcx.expect_def(pat.id));

                let subpatterns =
                        subpatterns.iter()
                                   .enumerate_and_adjust(variant_def.fields.len(), ddpos)
                                   .map(|(i, field)| FieldPattern {
                                       field: Field::new(i),
                                       pattern: self.to_pattern(field),
                                   })
                                   .collect();
                self.variant_or_leaf(pat, subpatterns)
            }

            PatKind::Struct(_, ref fields, _) => {
                let pat_ty = self.cx.tcx.node_id_to_type(pat.id);
                let adt_def = match pat_ty.sty {
                    ty::TyStruct(adt_def, _) | ty::TyEnum(adt_def, _) => adt_def,
                    _ => {
                        span_bug!(
                            pat.span,
                            "struct pattern not applied to struct or enum");
                    }
                };
                let variant_def = adt_def.variant_of_def(self.cx.tcx.expect_def(pat.id));

                let subpatterns =
                    fields.iter()
                          .map(|field| {
                              let index = variant_def.index_of_field_named(field.node.name);
                              let index = index.unwrap_or_else(|| {
                                  span_bug!(
                                      pat.span,
                                      "no field with name {:?}",
                                      field.node.name);
                              });
                              FieldPattern {
                                  field: Field::new(index),
                                  pattern: self.to_pattern(&field.node.pat),
                              }
                          })
                          .collect();

                self.variant_or_leaf(pat, subpatterns)
            }

            PatKind::QPath(..) => {
                span_bug!(pat.span, "unexpanded macro or bad constant etc");
            }
        };

        Pattern {
            span: pat.span,
            ty: ty,
            kind: Box::new(kind),
        }
    }

    fn to_patterns(&mut self, pats: &[P<hir::Pat>]) -> Vec<Pattern<'tcx>> {
        pats.iter().map(|p| self.to_pattern(p)).collect()
    }

    fn to_opt_pattern(&mut self, pat: &Option<P<hir::Pat>>) -> Option<Pattern<'tcx>> {
        pat.as_ref().map(|p| self.to_pattern(p))
    }

    fn slice_or_array_pattern(&mut self,
                              span: Span,
                              ty: Ty<'tcx>,
                              prefix: &[P<hir::Pat>],
                              slice: &Option<P<hir::Pat>>,
                              suffix: &[P<hir::Pat>])
                              -> PatternKind<'tcx> {
        match ty.sty {
            ty::TySlice(..) => {
                // matching a slice or fixed-length array
                PatternKind::Slice {
                    prefix: self.to_patterns(prefix),
                    slice: self.to_opt_pattern(slice),
                    suffix: self.to_patterns(suffix),
                }
            }

            ty::TyArray(_, len) => {
                // fixed-length array
                assert!(len >= prefix.len() + suffix.len());
                PatternKind::Array {
                    prefix: self.to_patterns(prefix),
                    slice: self.to_opt_pattern(slice),
                    suffix: self.to_patterns(suffix),
                }
            }

            _ => {
                span_bug!(span, "unexpanded macro or bad constant etc");
            }
        }
    }

    fn variant_or_leaf(&mut self,
                       pat: &hir::Pat,
                       subpatterns: Vec<FieldPattern<'tcx>>)
                       -> PatternKind<'tcx> {
        match self.cx.tcx.expect_def(pat.id) {
            Def::Variant(enum_id, variant_id) => {
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

            Def::Struct(..) | Def::TyAlias(..) => {
                PatternKind::Leaf { subpatterns: subpatterns }
            }

            def => {
                span_bug!(pat.span, "inappropriate def for pattern: {:?}", def);
            }
        }
    }
}
