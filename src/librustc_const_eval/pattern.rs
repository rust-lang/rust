// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use eval;

use rustc::middle::const_val::ConstVal;
use rustc::mir::repr::{Field, Literal, BorrowKind, Mutability};
use rustc::ty::{self, TyCtxt, AdtDef, Ty, Region};
use rustc::hir::{self, PatKind};
use rustc::hir::def::Def;
use rustc::hir::pat_util::EnumerateAndAdjustIterator;

use rustc_data_structures::indexed_vec::Idx;

use syntax::ast;
use syntax::ptr::P;
use syntax_pos::Span;

#[derive(Copy, Clone, Debug)]
pub enum BindingMode<'tcx> {
    ByValue,
    ByRef(&'tcx Region, BorrowKind),
}

#[derive(Clone, Debug)]
pub struct FieldPattern<'tcx> {
    pub field: Field,
    pub pattern: Pattern<'tcx>,
}

#[derive(Clone, Debug)]
pub struct Pattern<'tcx> {
    pub ty: Ty<'tcx>,
    pub span: Span,
    pub kind: Box<PatternKind<'tcx>>,
}

#[derive(Clone, Debug)]
pub enum PatternKind<'tcx> {
    Wild,

    /// x, ref x, x @ P, etc
    Binding {
        mutability: Mutability,
        name: ast::Name,
        mode: BindingMode<'tcx>,
        var: ast::NodeId,
        ty: Ty<'tcx>,
        subpattern: Option<Pattern<'tcx>>,
    },

    /// Foo(...) or Foo{...} or Foo, where `Foo` is a variant name from an adt with >1 variants
    Variant {
        adt_def: AdtDef<'tcx>,
        variant_index: usize,
        subpatterns: Vec<FieldPattern<'tcx>>,
    },

    /// (...), Foo(...), Foo{...}, or Foo, where `Foo` is a variant name from an adt with 1 variant
    Leaf {
        subpatterns: Vec<FieldPattern<'tcx>>,
    },

    /// box P, &P, &mut P, etc
    Deref {
        subpattern: Pattern<'tcx>,
    },

    Constant {
        value: ConstVal,
    },

    Range {
        lo: Literal<'tcx>,
        hi: Literal<'tcx>,
    },

    /// matches against a slice, checking the length and extracting elements
    Slice {
        prefix: Vec<Pattern<'tcx>>,
        slice: Option<Pattern<'tcx>>,
        suffix: Vec<Pattern<'tcx>>,
    },

    /// fixed match against an array, irrefutable
    Array {
        prefix: Vec<Pattern<'tcx>>,
        slice: Option<Pattern<'tcx>>,
        suffix: Vec<Pattern<'tcx>>,
    },
}

impl<'a, 'gcx, 'tcx> Pattern<'tcx> {
    pub fn from_hir(tcx: TyCtxt<'a, 'gcx, 'tcx>, pat: &hir::Pat) -> Self {
        let mut ty = tcx.node_id_to_type(pat.id);

        let kind = match pat.node {
            PatKind::Wild => PatternKind::Wild,

            PatKind::Lit(ref value) => {
                let value = eval::eval_const_expr(tcx.global_tcx(), value);
                PatternKind::Constant { value: value }
            }

            PatKind::Range(ref lo, ref hi) => {
                let lo = eval::eval_const_expr(tcx.global_tcx(), lo);
                let lo = Literal::Value { value: lo };
                let hi = eval::eval_const_expr(tcx.global_tcx(), hi);
                let hi = Literal::Value { value: hi };
                PatternKind::Range { lo: lo, hi: hi }
            },

            PatKind::Path(..) => {
                match tcx.expect_def(pat.id) {
                    Def::Const(def_id) | Def::AssociatedConst(def_id) => {
                        let tcx = tcx.global_tcx();
                        let substs = Some(tcx.node_id_item_substs(pat.id).substs);
                        match eval::lookup_const_by_id(tcx, def_id, substs) {
                            Some((const_expr, _const_ty)) => {
                                match eval::const_expr_to_pat(tcx,
                                                              const_expr,
                                                              pat.id,
                                                              pat.span) {
                                    Ok(pat) =>
                                        return Pattern::from_hir(tcx, &pat),
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
                    _ => {
                        PatternKind::from_variant_or_leaf(tcx, pat, vec![])
                    }
                }
            }

            PatKind::Ref(ref subpattern, _) |
            PatKind::Box(ref subpattern) => {
                PatternKind::Deref { subpattern: Self::from_hir(tcx, subpattern) }
            }

            PatKind::Slice(ref prefix, ref slice, ref suffix) => {
                let ty = tcx.node_id_to_type(pat.id);
                match ty.sty {
                    ty::TyRef(_, mt) =>
                        PatternKind::Deref {
                            subpattern: Pattern {
                                ty: mt.ty,
                                span: pat.span,
                                kind: Box::new(PatternKind::from_slice_or_array(
                                    tcx, pat.span, mt.ty, prefix, slice, suffix))
                            },
                        },

                    ty::TySlice(..) |
                    ty::TyArray(..) =>
                        PatternKind::from_slice_or_array(
                            tcx, pat.span, ty, prefix, slice, suffix),

                    ref sty =>
                        span_bug!(
                            pat.span,
                            "unexpanded type for vector pattern: {:?}",
                            sty),
                }
            }

            PatKind::Tuple(ref subpatterns, ddpos) => {
                match tcx.node_id_to_type(pat.id).sty {
                    ty::TyTuple(ref tys) => {
                        let subpatterns =
                            subpatterns.iter()
                                       .enumerate_and_adjust(tys.len(), ddpos)
                                       .map(|(i, subpattern)| FieldPattern {
                                            field: Field::new(i),
                                            pattern: Self::from_hir(tcx, subpattern),
                                       })
                                       .collect();

                        PatternKind::Leaf { subpatterns: subpatterns }
                    }

                    ref sty => span_bug!(pat.span, "unexpected type for tuple pattern: {:?}", sty),
                }
            }

            PatKind::Binding(bm, ref ident, ref sub) => {
                let def_id = tcx.expect_def(pat.id).def_id();
                let id = tcx.map.as_local_node_id(def_id).unwrap();
                let var_ty = tcx.node_id_to_type(pat.id);
                let region = match var_ty.sty {
                    ty::TyRef(r, _) => Some(r),
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
                    subpattern: Self::from_opt_pattern(tcx, sub),
                }
            }

            PatKind::TupleStruct(_, ref subpatterns, ddpos) => {
                let pat_ty = tcx.node_id_to_type(pat.id);
                let adt_def = match pat_ty.sty {
                    ty::TyAdt(adt_def, _) => adt_def,
                    _ => span_bug!(pat.span, "tuple struct pattern not applied to an ADT"),
                };
                let variant_def = adt_def.variant_of_def(tcx.expect_def(pat.id));

                let subpatterns =
                        subpatterns.iter()
                                   .enumerate_and_adjust(variant_def.fields.len(), ddpos)
                                   .map(|(i, field)| FieldPattern {
                                       field: Field::new(i),
                                       pattern: Self::from_hir(tcx, field),
                                   })
                                   .collect();
                PatternKind::from_variant_or_leaf(tcx, pat, subpatterns)
            }

            PatKind::Struct(_, ref fields, _) => {
                let pat_ty = tcx.node_id_to_type(pat.id);
                let adt_def = match pat_ty.sty {
                    ty::TyAdt(adt_def, _) => adt_def,
                    _ => {
                        span_bug!(
                            pat.span,
                            "struct pattern not applied to an ADT");
                    }
                };
                let variant_def = adt_def.variant_of_def(tcx.expect_def(pat.id));

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
                                  pattern: Self::from_hir(tcx, &field.node.pat),
                              }
                          })
                          .collect();

                PatternKind::from_variant_or_leaf(tcx, pat, subpatterns)
            }
        };

        Pattern {
            span: pat.span,
            ty: ty,
            kind: Box::new(kind),
        }
    }

    fn from_patterns(tcx: TyCtxt<'a, 'gcx, 'tcx>, pats: &[P<hir::Pat>]) -> Vec<Self> {
        pats.iter().map(|p| Self::from_hir(tcx, p)).collect()
    }

    fn from_opt_pattern(tcx: TyCtxt<'a, 'gcx, 'tcx>, pat: &Option<P<hir::Pat>>) -> Option<Self>
    {
        pat.as_ref().map(|p| Self::from_hir(tcx, p))
    }
}

impl<'a, 'gcx, 'tcx> PatternKind<'tcx> {
    fn from_slice_or_array(
        tcx: TyCtxt<'a, 'gcx, 'tcx>,
        span: Span,
        ty: Ty<'tcx>,
        prefix: &[P<hir::Pat>],
        slice: &Option<P<hir::Pat>>,
        suffix: &[P<hir::Pat>])
        -> Self
    {
        match ty.sty {
            ty::TySlice(..) => {
                // matching a slice or fixed-length array
                PatternKind::Slice {
                    prefix: Pattern::from_patterns(tcx, prefix),
                    slice: Pattern::from_opt_pattern(tcx, slice),
                    suffix: Pattern::from_patterns(tcx, suffix),
                }
            }

            ty::TyArray(_, len) => {
                // fixed-length array
                assert!(len >= prefix.len() + suffix.len());
                PatternKind::Array {
                    prefix: Pattern::from_patterns(tcx, prefix),
                    slice: Pattern::from_opt_pattern(tcx, slice),
                    suffix: Pattern::from_patterns(tcx, suffix),
                }
            }

            _ => {
                span_bug!(span, "unexpanded macro or bad constant etc");
            }
        }
    }

    fn from_variant_or_leaf(
        tcx: TyCtxt<'a, 'gcx, 'tcx>,
        pat: &hir::Pat,
        subpatterns: Vec<FieldPattern<'tcx>>)
        -> Self
    {
        match tcx.expect_def(pat.id) {
            Def::Variant(variant_id) | Def::VariantCtor(variant_id, ..) => {
                let enum_id = tcx.parent_def_id(variant_id).unwrap();
                let adt_def = tcx.lookup_adt_def(enum_id);
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

            Def::Struct(..) | Def::StructCtor(..) | Def::Union(..) |
            Def::TyAlias(..) | Def::AssociatedTy(..) => {
                PatternKind::Leaf { subpatterns: subpatterns }
            }

            def => {
                span_bug!(pat.span, "inappropriate def for pattern: {:?}", def);
            }
        }
    }
}
