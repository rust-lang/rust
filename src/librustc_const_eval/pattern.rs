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

use rustc::middle::const_val::{ConstEvalErr, ConstVal};
use rustc::mir::{Field, BorrowKind, Mutability};
use rustc::ty::{self, TyCtxt, AdtDef, Ty, Region};
use rustc::ty::subst::{Substs, Kind};
use rustc::hir::{self, PatKind, RangeEnd};
use rustc::hir::def::{Def, CtorKind};
use rustc::hir::pat_util::EnumerateAndAdjustIterator;

use rustc_data_structures::indexed_vec::Idx;

use std::fmt;
use syntax::ast;
use syntax::ptr::P;
use syntax_pos::Span;

#[derive(Clone, Debug)]
pub enum PatternError<'tcx> {
    StaticInPattern(Span),
    ConstEval(ConstEvalErr<'tcx>),
}

#[derive(Copy, Clone, Debug)]
pub enum BindingMode<'tcx> {
    ByValue,
    ByRef(Region<'tcx>, BorrowKind),
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
        adt_def: &'tcx AdtDef,
        substs: &'tcx Substs<'tcx>,
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
        value: &'tcx ty::Const<'tcx>,
    },

    Range {
        lo: &'tcx ty::Const<'tcx>,
        hi: &'tcx ty::Const<'tcx>,
        end: RangeEnd,
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

fn print_const_val(value: &ConstVal, f: &mut fmt::Formatter) -> fmt::Result {
    match *value {
        ConstVal::Float(ref x) => write!(f, "{}", x),
        ConstVal::Integral(ref i) => write!(f, "{}", i),
        ConstVal::Str(ref s) => write!(f, "{:?}", &s[..]),
        ConstVal::ByteStr(b) => write!(f, "{:?}", b.data),
        ConstVal::Bool(b) => write!(f, "{:?}", b),
        ConstVal::Char(c) => write!(f, "{:?}", c),
        ConstVal::Variant(_) |
        ConstVal::Function(..) |
        ConstVal::Aggregate(_) |
        ConstVal::Unevaluated(..) => bug!("{:?} not printable in a pattern", value)
    }
}

impl<'tcx> fmt::Display for Pattern<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self.kind {
            PatternKind::Wild => write!(f, "_"),
            PatternKind::Binding { mutability, name, mode, ref subpattern, .. } => {
                let is_mut = match mode {
                    BindingMode::ByValue => mutability == Mutability::Mut,
                    BindingMode::ByRef(_, bk) => {
                        write!(f, "ref ")?;
                        bk == BorrowKind::Mut
                    }
                };
                if is_mut {
                    write!(f, "mut ")?;
                }
                write!(f, "{}", name)?;
                if let Some(ref subpattern) = *subpattern {
                    write!(f, " @ {}", subpattern)?;
                }
                Ok(())
            }
            PatternKind::Variant { ref subpatterns, .. } |
            PatternKind::Leaf { ref subpatterns } => {
                let variant = match *self.kind {
                    PatternKind::Variant { adt_def, variant_index, .. } => {
                        Some(&adt_def.variants[variant_index])
                    }
                    _ => if let ty::TyAdt(adt, _) = self.ty.sty {
                        if adt.is_univariant() {
                            Some(&adt.variants[0])
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                };

                let mut first = true;
                let mut start_or_continue = || if first { first = false; "" } else { ", " };

                if let Some(variant) = variant {
                    write!(f, "{}", variant.name)?;

                    // Only for TyAdt we can have `S {...}`,
                    // which we handle separately here.
                    if variant.ctor_kind == CtorKind::Fictive {
                        write!(f, " {{ ")?;

                        let mut printed = 0;
                        for p in subpatterns {
                            if let PatternKind::Wild = *p.pattern.kind {
                                continue;
                            }
                            let name = variant.fields[p.field.index()].name;
                            write!(f, "{}{}: {}", start_or_continue(), name, p.pattern)?;
                            printed += 1;
                        }

                        if printed < variant.fields.len() {
                            write!(f, "{}..", start_or_continue())?;
                        }

                        return write!(f, " }}");
                    }
                }

                let num_fields = variant.map_or(subpatterns.len(), |v| v.fields.len());
                if num_fields != 0 || variant.is_none() {
                    write!(f, "(")?;
                    for i in 0..num_fields {
                        write!(f, "{}", start_or_continue())?;

                        // Common case: the field is where we expect it.
                        if let Some(p) = subpatterns.get(i) {
                            if p.field.index() == i {
                                write!(f, "{}", p.pattern)?;
                                continue;
                            }
                        }

                        // Otherwise, we have to go looking for it.
                        if let Some(p) = subpatterns.iter().find(|p| p.field.index() == i) {
                            write!(f, "{}", p.pattern)?;
                        } else {
                            write!(f, "_")?;
                        }
                    }
                    write!(f, ")")?;
                }

                Ok(())
            }
            PatternKind::Deref { ref subpattern } => {
                match self.ty.sty {
                    ty::TyAdt(def, _) if def.is_box() => write!(f, "box ")?,
                    ty::TyRef(_, mt) => {
                        write!(f, "&")?;
                        if mt.mutbl == hir::MutMutable {
                            write!(f, "mut ")?;
                        }
                    }
                    _ => bug!("{} is a bad Deref pattern type", self.ty)
                }
                write!(f, "{}", subpattern)
            }
            PatternKind::Constant { value } => {
                print_const_val(&value.val, f)
            }
            PatternKind::Range { lo, hi, end } => {
                print_const_val(&lo.val, f)?;
                match end {
                    RangeEnd::Included => write!(f, "...")?,
                    RangeEnd::Excluded => write!(f, "..")?,
                }
                print_const_val(&hi.val, f)
            }
            PatternKind::Slice { ref prefix, ref slice, ref suffix } |
            PatternKind::Array { ref prefix, ref slice, ref suffix } => {
                let mut first = true;
                let mut start_or_continue = || if first { first = false; "" } else { ", " };
                write!(f, "[")?;
                for p in prefix {
                    write!(f, "{}{}", start_or_continue(), p)?;
                }
                if let Some(ref slice) = *slice {
                    write!(f, "{}", start_or_continue())?;
                    match *slice.kind {
                        PatternKind::Wild => {}
                        _ => write!(f, "{}", slice)?
                    }
                    write!(f, "..")?;
                }
                for p in suffix {
                    write!(f, "{}{}", start_or_continue(), p)?;
                }
                write!(f, "]")
            }
        }
    }
}

pub struct PatternContext<'a, 'tcx: 'a> {
    pub tcx: TyCtxt<'a, 'tcx, 'tcx>,
    pub param_env: ty::ParamEnv<'tcx>,
    pub tables: &'a ty::TypeckTables<'tcx>,
    pub substs: &'tcx Substs<'tcx>,
    pub errors: Vec<PatternError<'tcx>>,
}

impl<'a, 'tcx> Pattern<'tcx> {
    pub fn from_hir(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                    param_env_and_substs: ty::ParamEnvAnd<'tcx, &'tcx Substs<'tcx>>,
                    tables: &'a ty::TypeckTables<'tcx>,
                    pat: &'tcx hir::Pat) -> Self {
        let mut pcx = PatternContext::new(tcx, param_env_and_substs, tables);
        let result = pcx.lower_pattern(pat);
        if !pcx.errors.is_empty() {
            span_bug!(pat.span, "encountered errors lowering pattern: {:?}", pcx.errors)
        }
        debug!("Pattern::from_hir({:?}) = {:?}", pat, result);
        result
    }
}

impl<'a, 'tcx> PatternContext<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>,
               param_env_and_substs: ty::ParamEnvAnd<'tcx, &'tcx Substs<'tcx>>,
               tables: &'a ty::TypeckTables<'tcx>) -> Self {
        PatternContext {
            tcx,
            param_env: param_env_and_substs.param_env,
            tables,
            substs: param_env_and_substs.value,
            errors: vec![]
        }
    }

    pub fn lower_pattern(&mut self, pat: &'tcx hir::Pat) -> Pattern<'tcx> {
        let mut ty = self.tables.node_id_to_type(pat.hir_id);

        let kind = match pat.node {
            PatKind::Wild => PatternKind::Wild,

            PatKind::Lit(ref value) => self.lower_lit(value),

            PatKind::Range(ref lo, ref hi, end) => {
                match (self.lower_lit(lo), self.lower_lit(hi)) {
                    (PatternKind::Constant { value: lo },
                     PatternKind::Constant { value: hi }) => {
                        PatternKind::Range { lo, hi, end }
                    }
                    _ => PatternKind::Wild
                }
            }

            PatKind::Path(ref qpath) => {
                return self.lower_path(qpath, pat.hir_id, pat.id, pat.span);
            }

            PatKind::Ref(ref subpattern, _) |
            PatKind::Box(ref subpattern) => {
                PatternKind::Deref { subpattern: self.lower_pattern(subpattern) }
            }

            PatKind::Slice(ref prefix, ref slice, ref suffix) => {
                let ty = self.tables.node_id_to_type(pat.hir_id);
                match ty.sty {
                    ty::TyRef(_, mt) =>
                        PatternKind::Deref {
                            subpattern: Pattern {
                                ty: mt.ty,
                                span: pat.span,
                                kind: Box::new(self.slice_or_array_pattern(
                                    pat.span, mt.ty, prefix, slice, suffix))
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
                let ty = self.tables.node_id_to_type(pat.hir_id);
                match ty.sty {
                    ty::TyTuple(ref tys, _) => {
                        let subpatterns =
                            subpatterns.iter()
                                       .enumerate_and_adjust(tys.len(), ddpos)
                                       .map(|(i, subpattern)| FieldPattern {
                                            field: Field::new(i),
                                            pattern: self.lower_pattern(subpattern)
                                       })
                                       .collect();

                        PatternKind::Leaf { subpatterns: subpatterns }
                    }

                    ref sty => span_bug!(pat.span, "unexpected type for tuple pattern: {:?}", sty),
                }
            }

            PatKind::Binding(_, id, ref ident, ref sub) => {
                let var_ty = self.tables.node_id_to_type(pat.hir_id);
                let region = match var_ty.sty {
                    ty::TyRef(r, _) => Some(r),
                    _ => None,
                };
                let bm = *self.tables.pat_binding_modes().get(pat.hir_id)
                                                         .expect("missing binding mode");
                let (mutability, mode) = match bm {
                    ty::BindByValue(hir::MutMutable) =>
                        (Mutability::Mut, BindingMode::ByValue),
                    ty::BindByValue(hir::MutImmutable) =>
                        (Mutability::Not, BindingMode::ByValue),
                    ty::BindByReference(hir::MutMutable) =>
                        (Mutability::Not, BindingMode::ByRef(
                            region.unwrap(), BorrowKind::Mut)),
                    ty::BindByReference(hir::MutImmutable) =>
                        (Mutability::Not, BindingMode::ByRef(
                            region.unwrap(), BorrowKind::Shared)),
                };

                // A ref x pattern is the same node used for x, and as such it has
                // x's type, which is &T, where we want T (the type being matched).
                if let ty::BindByReference(_) = bm {
                    if let ty::TyRef(_, mt) = ty.sty {
                        ty = mt.ty;
                    } else {
                        bug!("`ref {}` has wrong type {}", ident.node, ty);
                    }
                }

                PatternKind::Binding {
                    mutability,
                    mode,
                    name: ident.node,
                    var: id,
                    ty: var_ty,
                    subpattern: self.lower_opt_pattern(sub),
                }
            }

            PatKind::TupleStruct(ref qpath, ref subpatterns, ddpos) => {
                let def = self.tables.qpath_def(qpath, pat.hir_id);
                let adt_def = match ty.sty {
                    ty::TyAdt(adt_def, _) => adt_def,
                    _ => span_bug!(pat.span, "tuple struct pattern not applied to an ADT"),
                };
                let variant_def = adt_def.variant_of_def(def);

                let subpatterns =
                        subpatterns.iter()
                                   .enumerate_and_adjust(variant_def.fields.len(), ddpos)
                                   .map(|(i, field)| FieldPattern {
                                       field: Field::new(i),
                                       pattern: self.lower_pattern(field),
                                   })
                                   .collect();
                self.lower_variant_or_leaf(def, ty, subpatterns)
            }

            PatKind::Struct(ref qpath, ref fields, _) => {
                let def = self.tables.qpath_def(qpath, pat.hir_id);
                let adt_def = match ty.sty {
                    ty::TyAdt(adt_def, _) => adt_def,
                    _ => {
                        span_bug!(
                            pat.span,
                            "struct pattern not applied to an ADT");
                    }
                };
                let variant_def = adt_def.variant_of_def(def);

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
                                  pattern: self.lower_pattern(&field.node.pat),
                              }
                          })
                          .collect();

                self.lower_variant_or_leaf(def, ty, subpatterns)
            }
        };

        Pattern {
            span: pat.span,
            ty,
            kind: Box::new(kind),
        }
    }

    fn lower_patterns(&mut self, pats: &'tcx [P<hir::Pat>]) -> Vec<Pattern<'tcx>> {
        pats.iter().map(|p| self.lower_pattern(p)).collect()
    }

    fn lower_opt_pattern(&mut self, pat: &'tcx Option<P<hir::Pat>>) -> Option<Pattern<'tcx>>
    {
        pat.as_ref().map(|p| self.lower_pattern(p))
    }

    fn flatten_nested_slice_patterns(
        &mut self,
        prefix: Vec<Pattern<'tcx>>,
        slice: Option<Pattern<'tcx>>,
        suffix: Vec<Pattern<'tcx>>)
        -> (Vec<Pattern<'tcx>>, Option<Pattern<'tcx>>, Vec<Pattern<'tcx>>)
    {
        let orig_slice = match slice {
            Some(orig_slice) => orig_slice,
            None => return (prefix, slice, suffix)
        };
        let orig_prefix = prefix;
        let orig_suffix = suffix;

        // dance because of intentional borrow-checker stupidity.
        let kind = *orig_slice.kind;
        match kind {
            PatternKind::Slice { prefix, slice, mut suffix } |
            PatternKind::Array { prefix, slice, mut suffix } => {
                let mut orig_prefix = orig_prefix;

                orig_prefix.extend(prefix);
                suffix.extend(orig_suffix);

                (orig_prefix, slice, suffix)
            }
            _ => {
                (orig_prefix, Some(Pattern {
                    kind: box kind, ..orig_slice
                }), orig_suffix)
            }
        }
    }

    fn slice_or_array_pattern(
        &mut self,
        span: Span,
        ty: Ty<'tcx>,
        prefix: &'tcx [P<hir::Pat>],
        slice: &'tcx Option<P<hir::Pat>>,
        suffix: &'tcx [P<hir::Pat>])
        -> PatternKind<'tcx>
    {
        let prefix = self.lower_patterns(prefix);
        let slice = self.lower_opt_pattern(slice);
        let suffix = self.lower_patterns(suffix);
        let (prefix, slice, suffix) =
            self.flatten_nested_slice_patterns(prefix, slice, suffix);

        match ty.sty {
            ty::TySlice(..) => {
                // matching a slice or fixed-length array
                PatternKind::Slice { prefix: prefix, slice: slice, suffix: suffix }
            }

            ty::TyArray(_, len) => {
                // fixed-length array
                let len = len.val.to_const_int().unwrap().to_u64().unwrap();
                assert!(len >= prefix.len() as u64 + suffix.len() as u64);
                PatternKind::Array { prefix: prefix, slice: slice, suffix: suffix }
            }

            _ => {
                span_bug!(span, "bad slice pattern type {:?}", ty);
            }
        }
    }

    fn lower_variant_or_leaf(
        &mut self,
        def: Def,
        ty: Ty<'tcx>,
        subpatterns: Vec<FieldPattern<'tcx>>)
        -> PatternKind<'tcx>
    {
        match def {
            Def::Variant(variant_id) | Def::VariantCtor(variant_id, ..) => {
                let enum_id = self.tcx.parent_def_id(variant_id).unwrap();
                let adt_def = self.tcx.adt_def(enum_id);
                if adt_def.variants.len() > 1 {
                    let substs = match ty.sty {
                        ty::TyAdt(_, substs) |
                        ty::TyFnDef(_, substs) => substs,
                        _ => bug!("inappropriate type for def: {:?}", ty.sty),
                    };
                    PatternKind::Variant {
                        adt_def,
                        substs,
                        variant_index: adt_def.variant_index_with_id(variant_id),
                        subpatterns,
                    }
                } else {
                    PatternKind::Leaf { subpatterns: subpatterns }
                }
            }

            Def::Struct(..) | Def::StructCtor(..) | Def::Union(..) |
            Def::TyAlias(..) | Def::AssociatedTy(..) | Def::SelfTy(..) => {
                PatternKind::Leaf { subpatterns: subpatterns }
            }

            _ => bug!()
        }
    }

    fn lower_path(&mut self,
                  qpath: &hir::QPath,
                  id: hir::HirId,
                  pat_id: ast::NodeId,
                  span: Span)
                  -> Pattern<'tcx> {
        let ty = self.tables.node_id_to_type(id);
        let def = self.tables.qpath_def(qpath, id);
        let kind = match def {
            Def::Const(def_id) | Def::AssociatedConst(def_id) => {
                let substs = self.tables.node_substs(id);
                match eval::lookup_const_by_id(self.tcx, self.param_env.and((def_id, substs))) {
                    Some((def_id, substs)) => {
                        // Enter the inlined constant's tables&substs temporarily.
                        let old_tables = self.tables;
                        let old_substs = self.substs;
                        self.tables = self.tcx.typeck_tables_of(def_id);
                        self.substs = substs;
                        let body = if let Some(id) = self.tcx.hir.as_local_node_id(def_id) {
                            self.tcx.hir.body(self.tcx.hir.body_owned_by(id))
                        } else {
                            self.tcx.extern_const_body(def_id).body
                        };
                        let pat = self.lower_const_expr(&body.value, pat_id, span);
                        self.tables = old_tables;
                        self.substs = old_substs;
                        return pat;
                    }
                    None => {
                        self.errors.push(PatternError::StaticInPattern(span));
                        PatternKind::Wild
                    }
                }
            }
            _ => self.lower_variant_or_leaf(def, ty, vec![]),
        };

        Pattern {
            span,
            ty,
            kind: Box::new(kind),
        }
    }

    fn lower_lit(&mut self, expr: &'tcx hir::Expr) -> PatternKind<'tcx> {
        let const_cx = eval::ConstContext::new(self.tcx,
                                               self.param_env.and(self.substs),
                                               self.tables);
        match const_cx.eval(expr) {
            Ok(value) => {
                if let ConstVal::Variant(def_id) = value.val {
                    let ty = self.tables.expr_ty(expr);
                    self.lower_variant_or_leaf(Def::Variant(def_id), ty, vec![])
                } else {
                    PatternKind::Constant { value }
                }
            }
            Err(e) => {
                self.errors.push(PatternError::ConstEval(e));
                PatternKind::Wild
            }
        }
    }

    fn lower_const_expr(&mut self,
                        expr: &'tcx hir::Expr,
                        pat_id: ast::NodeId,
                        span: Span)
                        -> Pattern<'tcx> {
        let pat_ty = self.tables.expr_ty(expr);
        debug!("expr={:?} pat_ty={:?} pat_id={}", expr, pat_ty, pat_id);
        match pat_ty.sty {
            ty::TyFloat(_) => {
                self.tcx.sess.span_err(span, "floating point constants cannot be used in patterns");
            }
            ty::TyAdt(adt_def, _) if adt_def.is_union() => {
                // Matching on union fields is unsafe, we can't hide it in constants
                self.tcx.sess.span_err(span, "cannot use unions in constant patterns");
            }
            ty::TyAdt(adt_def, _) => {
                if !self.tcx.has_attr(adt_def.did, "structural_match") {
                    let msg = format!("to use a constant of type `{}` in a pattern, \
                                       `{}` must be annotated with `#[derive(PartialEq, Eq)]`",
                                      self.tcx.item_path_str(adt_def.did),
                                      self.tcx.item_path_str(adt_def.did));
                    self.tcx.sess.span_err(span, &msg);
                }
            }
            _ => { }
        }
        let kind = match expr.node {
            hir::ExprTup(ref exprs) => {
                PatternKind::Leaf {
                    subpatterns: exprs.iter().enumerate().map(|(i, expr)| {
                        FieldPattern {
                            field: Field::new(i),
                            pattern: self.lower_const_expr(expr, pat_id, span)
                        }
                    }).collect()
                }
            }

            hir::ExprCall(ref callee, ref args) => {
                let qpath = match callee.node {
                    hir::ExprPath(ref qpath) => qpath,
                    _ => bug!()
                };
                let ty = self.tables.node_id_to_type(callee.hir_id);
                let def = self.tables.qpath_def(qpath, callee.hir_id);
                match def {
                    Def::Fn(..) | Def::Method(..) => self.lower_lit(expr),
                    _ => {
                        let subpatterns = args.iter().enumerate().map(|(i, expr)| {
                            FieldPattern {
                                field: Field::new(i),
                                pattern: self.lower_const_expr(expr, pat_id, span)
                            }
                        }).collect();
                        self.lower_variant_or_leaf(def, ty, subpatterns)
                    }
                }
            }

            hir::ExprStruct(ref qpath, ref fields, None) => {
                let def = self.tables.qpath_def(qpath, expr.hir_id);
                let adt_def = match pat_ty.sty {
                    ty::TyAdt(adt_def, _) => adt_def,
                    _ => {
                        span_bug!(
                            expr.span,
                            "struct expr without ADT type");
                    }
                };
                let variant_def = adt_def.variant_of_def(def);

                let subpatterns =
                    fields.iter()
                          .map(|field| {
                              let index = variant_def.index_of_field_named(field.name.node);
                              let index = index.unwrap_or_else(|| {
                                  span_bug!(
                                      expr.span,
                                      "no field with name {:?}",
                                      field.name);
                              });
                              FieldPattern {
                                  field: Field::new(index),
                                  pattern: self.lower_const_expr(&field.expr, pat_id, span),
                              }
                          })
                          .collect();

                self.lower_variant_or_leaf(def, pat_ty, subpatterns)
            }

            hir::ExprArray(ref exprs) => {
                let pats = exprs.iter()
                                .map(|expr| self.lower_const_expr(expr, pat_id, span))
                                .collect();
                PatternKind::Array {
                    prefix: pats,
                    slice: None,
                    suffix: vec![]
                }
            }

            hir::ExprPath(ref qpath) => {
                return self.lower_path(qpath, expr.hir_id, pat_id, span);
            }

            _ => self.lower_lit(expr)
        };

        Pattern {
            span,
            ty: pat_ty,
            kind: Box::new(kind),
        }
    }
}

pub trait PatternFoldable<'tcx> : Sized {
    fn fold_with<F: PatternFolder<'tcx>>(&self, folder: &mut F) -> Self {
        self.super_fold_with(folder)
    }

    fn super_fold_with<F: PatternFolder<'tcx>>(&self, folder: &mut F) -> Self;
}

pub trait PatternFolder<'tcx> : Sized {
    fn fold_pattern(&mut self, pattern: &Pattern<'tcx>) -> Pattern<'tcx> {
        pattern.super_fold_with(self)
    }

    fn fold_pattern_kind(&mut self, kind: &PatternKind<'tcx>) -> PatternKind<'tcx> {
        kind.super_fold_with(self)
    }
}


impl<'tcx, T: PatternFoldable<'tcx>> PatternFoldable<'tcx> for Box<T> {
    fn super_fold_with<F: PatternFolder<'tcx>>(&self, folder: &mut F) -> Self {
        let content: T = (**self).fold_with(folder);
        box content
    }
}

impl<'tcx, T: PatternFoldable<'tcx>> PatternFoldable<'tcx> for Vec<T> {
    fn super_fold_with<F: PatternFolder<'tcx>>(&self, folder: &mut F) -> Self {
        self.iter().map(|t| t.fold_with(folder)).collect()
    }
}

impl<'tcx, T: PatternFoldable<'tcx>> PatternFoldable<'tcx> for Option<T> {
    fn super_fold_with<F: PatternFolder<'tcx>>(&self, folder: &mut F) -> Self{
        self.as_ref().map(|t| t.fold_with(folder))
    }
}

macro_rules! CloneImpls {
    (<$lt_tcx:tt> $($ty:ty),+) => {
        $(
            impl<$lt_tcx> PatternFoldable<$lt_tcx> for $ty {
                fn super_fold_with<F: PatternFolder<$lt_tcx>>(&self, _: &mut F) -> Self {
                    Clone::clone(self)
                }
            }
        )+
    }
}

CloneImpls!{ <'tcx>
    Span, Field, Mutability, ast::Name, ast::NodeId, usize, &'tcx ty::Const<'tcx>,
    Region<'tcx>, Ty<'tcx>, BindingMode<'tcx>, &'tcx AdtDef,
    &'tcx Substs<'tcx>, &'tcx Kind<'tcx>
}

impl<'tcx> PatternFoldable<'tcx> for FieldPattern<'tcx> {
    fn super_fold_with<F: PatternFolder<'tcx>>(&self, folder: &mut F) -> Self {
        FieldPattern {
            field: self.field.fold_with(folder),
            pattern: self.pattern.fold_with(folder)
        }
    }
}

impl<'tcx> PatternFoldable<'tcx> for Pattern<'tcx> {
    fn fold_with<F: PatternFolder<'tcx>>(&self, folder: &mut F) -> Self {
        folder.fold_pattern(self)
    }

    fn super_fold_with<F: PatternFolder<'tcx>>(&self, folder: &mut F) -> Self {
        Pattern {
            ty: self.ty.fold_with(folder),
            span: self.span.fold_with(folder),
            kind: self.kind.fold_with(folder)
        }
    }
}

impl<'tcx> PatternFoldable<'tcx> for PatternKind<'tcx> {
    fn fold_with<F: PatternFolder<'tcx>>(&self, folder: &mut F) -> Self {
        folder.fold_pattern_kind(self)
    }

    fn super_fold_with<F: PatternFolder<'tcx>>(&self, folder: &mut F) -> Self {
        match *self {
            PatternKind::Wild => PatternKind::Wild,
            PatternKind::Binding {
                mutability,
                name,
                mode,
                var,
                ty,
                ref subpattern,
            } => PatternKind::Binding {
                mutability: mutability.fold_with(folder),
                name: name.fold_with(folder),
                mode: mode.fold_with(folder),
                var: var.fold_with(folder),
                ty: ty.fold_with(folder),
                subpattern: subpattern.fold_with(folder),
            },
            PatternKind::Variant {
                adt_def,
                substs,
                variant_index,
                ref subpatterns,
            } => PatternKind::Variant {
                adt_def: adt_def.fold_with(folder),
                substs: substs.fold_with(folder),
                variant_index: variant_index.fold_with(folder),
                subpatterns: subpatterns.fold_with(folder)
            },
            PatternKind::Leaf {
                ref subpatterns,
            } => PatternKind::Leaf {
                subpatterns: subpatterns.fold_with(folder),
            },
            PatternKind::Deref {
                ref subpattern,
            } => PatternKind::Deref {
                subpattern: subpattern.fold_with(folder),
            },
            PatternKind::Constant {
                value
            } => PatternKind::Constant {
                value: value.fold_with(folder)
            },
            PatternKind::Range {
                lo,
                hi,
                end,
            } => PatternKind::Range {
                lo: lo.fold_with(folder),
                hi: hi.fold_with(folder),
                end,
            },
            PatternKind::Slice {
                ref prefix,
                ref slice,
                ref suffix,
            } => PatternKind::Slice {
                prefix: prefix.fold_with(folder),
                slice: slice.fold_with(folder),
                suffix: suffix.fold_with(folder)
            },
            PatternKind::Array {
                ref prefix,
                ref slice,
                ref suffix
            } => PatternKind::Array {
                prefix: prefix.fold_with(folder),
                slice: slice.fold_with(folder),
                suffix: suffix.fold_with(folder)
            },
        }
    }
}
