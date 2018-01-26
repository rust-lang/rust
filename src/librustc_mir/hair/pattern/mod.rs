// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Code to validate patterns/matches

mod _match;
mod check_match;

pub use self::check_match::check_crate;
pub(crate) use self::check_match::check_match;

use interpret::{const_val_field, const_discr};

use rustc::middle::const_val::ConstVal;
use rustc::mir::{Field, BorrowKind, Mutability};
use rustc::mir::interpret::{GlobalId, Value, PrimVal};
use rustc::ty::{self, TyCtxt, AdtDef, Ty, Region};
use rustc::ty::subst::{Substs, Kind};
use rustc::hir::{self, PatKind, RangeEnd};
use rustc::hir::def::{Def, CtorKind};
use rustc::hir::pat_util::EnumerateAndAdjustIterator;

use rustc_data_structures::indexed_vec::Idx;
use rustc_const_math::ConstFloat;

use std::cmp::Ordering;
use std::fmt;
use syntax::ast;
use syntax::ptr::P;
use syntax_pos::Span;

#[derive(Clone, Debug)]
pub enum PatternError {
    StaticInPattern(Span),
    FloatBug,
    NonConstPath(Span),
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

    /// matches against a slice, checking the length and extracting elements.
    /// irrefutable when there is a slice pattern and both `prefix` and `suffix` are empty.
    /// e.g. `&[ref xs..]`.
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

fn print_const_val(value: &ty::Const, f: &mut fmt::Formatter) -> fmt::Result {
    match value.val {
        ConstVal::Value(v) => print_miri_value(v, value.ty, f),
        ConstVal::Unevaluated(..) => bug!("{:?} not printable in a pattern", value)
    }
}

fn print_miri_value(value: Value, ty: Ty, f: &mut fmt::Formatter) -> fmt::Result {
    use rustc::ty::TypeVariants::*;
    match (value, &ty.sty) {
        (Value::ByVal(PrimVal::Bytes(0)), &TyBool) => write!(f, "false"),
        (Value::ByVal(PrimVal::Bytes(1)), &TyBool) => write!(f, "true"),
        (Value::ByVal(PrimVal::Bytes(n)), &TyUint(..)) => write!(f, "{:?}", n),
        (Value::ByVal(PrimVal::Bytes(n)), &TyInt(..)) => write!(f, "{:?}", n as i128),
        (Value::ByVal(PrimVal::Bytes(n)), &TyChar) =>
            write!(f, "{:?}", ::std::char::from_u32(n as u32).unwrap()),
        _ => bug!("{:?}: {} not printable in a pattern", value, ty),
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
                        if !adt.is_enum() {
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
                print_const_val(value, f)
            }
            PatternKind::Range { lo, hi, end } => {
                print_const_val(lo, f)?;
                match end {
                    RangeEnd::Included => write!(f, "...")?,
                    RangeEnd::Excluded => write!(f, "..")?,
                }
                print_const_val(hi, f)
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
    pub errors: Vec<PatternError>,
}

impl<'a, 'tcx> Pattern<'tcx> {
    pub fn from_hir(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                    param_env_and_substs: ty::ParamEnvAnd<'tcx, &'tcx Substs<'tcx>>,
                    tables: &'a ty::TypeckTables<'tcx>,
                    pat: &'tcx hir::Pat) -> Self {
        let mut pcx = PatternContext::new(tcx, param_env_and_substs, tables);
        let result = pcx.lower_pattern(pat);
        if !pcx.errors.is_empty() {
            let msg = format!("encountered errors lowering pattern: {:?}", pcx.errors);
            tcx.sess.delay_span_bug(pat.span, &msg);
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
        // When implicit dereferences have been inserted in this pattern, the unadjusted lowered
        // pattern has the type that results *after* dereferencing. For example, in this code:
        //
        // ```
        // match &&Some(0i32) {
        //     Some(n) => { ... },
        //     _ => { ... },
        // }
        // ```
        //
        // the type assigned to `Some(n)` in `unadjusted_pat` would be `Option<i32>` (this is
        // determined in rustc_typeck::check::match). The adjustments would be
        //
        // `vec![&&Option<i32>, &Option<i32>]`.
        //
        // Applying the adjustments, we want to instead output `&&Some(n)` (as a HAIR pattern). So
        // we wrap the unadjusted pattern in `PatternKind::Deref` repeatedly, consuming the
        // adjustments in *reverse order* (last-in-first-out, so that the last `Deref` inserted
        // gets the least-dereferenced type).
        let unadjusted_pat = self.lower_pattern_unadjusted(pat);
        self.tables
            .pat_adjustments()
            .get(pat.hir_id)
            .unwrap_or(&vec![])
            .iter()
            .rev()
            .fold(unadjusted_pat, |pat, ref_ty| {
                    debug!("{:?}: wrapping pattern with type {:?}", pat, ref_ty);
                    Pattern {
                        span: pat.span,
                        ty: ref_ty,
                        kind: Box::new(PatternKind::Deref { subpattern: pat }),
                    }
                },
            )
    }

    fn lower_pattern_unadjusted(&mut self, pat: &'tcx hir::Pat) -> Pattern<'tcx> {
        let mut ty = self.tables.node_id_to_type(pat.hir_id);

        let kind = match pat.node {
            PatKind::Wild => PatternKind::Wild,

            PatKind::Lit(ref value) => self.lower_lit(value),

            PatKind::Range(ref lo_expr, ref hi_expr, end) => {
                match (self.lower_lit(lo_expr), self.lower_lit(hi_expr)) {
                    (PatternKind::Constant { value: lo },
                     PatternKind::Constant { value: hi }) => {
                        use std::cmp::Ordering;
                        match (end, compare_const_vals(&lo.val, &hi.val, ty).unwrap()) {
                            (RangeEnd::Excluded, Ordering::Less) => {},
                            (RangeEnd::Excluded, _) => span_err!(
                                self.tcx.sess,
                                lo_expr.span,
                                E0579,
                                "lower range bound must be less than upper",
                            ),
                            (RangeEnd::Included, Ordering::Greater) => {
                                struct_span_err!(self.tcx.sess, lo_expr.span, E0030,
                                    "lower range bound must be less than or equal to upper")
                                    .span_label(lo_expr.span, "lower bound larger than upper bound")
                                    .emit();
                            },
                            (RangeEnd::Included, _) => {}
                        }
                        PatternKind::Range { lo, hi, end }
                    }
                    _ => PatternKind::Wild
                }
            }

            PatKind::Path(ref qpath) => {
                return self.lower_path(qpath, pat.hir_id, pat.span);
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
                self.lower_variant_or_leaf(def, pat.span, ty, subpatterns)
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

                self.lower_variant_or_leaf(def, pat.span, ty, subpatterns)
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
                let len = len.val.unwrap_u64();
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
        span: Span,
        ty: Ty<'tcx>,
        subpatterns: Vec<FieldPattern<'tcx>>)
        -> PatternKind<'tcx>
    {
        match def {
            Def::Variant(variant_id) | Def::VariantCtor(variant_id, ..) => {
                let enum_id = self.tcx.parent_def_id(variant_id).unwrap();
                let adt_def = self.tcx.adt_def(enum_id);
                if adt_def.is_enum() {
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

            _ => {
                self.errors.push(PatternError::NonConstPath(span));
                PatternKind::Wild
            }
        }
    }

    fn lower_path(&mut self,
                  qpath: &hir::QPath,
                  id: hir::HirId,
                  span: Span)
                  -> Pattern<'tcx> {
        let ty = self.tables.node_id_to_type(id);
        let def = self.tables.qpath_def(qpath, id);
        let kind = match def {
            Def::Const(def_id) | Def::AssociatedConst(def_id) => {
                let substs = self.tables.node_substs(id);
                match ty::Instance::resolve(
                    self.tcx,
                    self.param_env,
                    def_id,
                    substs,
                ) {
                    Some(instance) => {
                        let cid = GlobalId {
                            instance,
                            promoted: None,
                        };
                        match self.tcx.at(span).const_eval(self.param_env.and(cid)) {
                            Ok(value) => {
                                return self.const_to_pat(instance, value, id, span)
                            },
                            Err(err) => {
                                err.report(self.tcx, span, "pattern");
                                PatternKind::Wild
                            },
                        }
                    },
                    None => {
                        self.errors.push(PatternError::StaticInPattern(span));
                        PatternKind::Wild
                    },
                }
            }
            _ => self.lower_variant_or_leaf(def, span, ty, vec![]),
        };

        Pattern {
            span,
            ty,
            kind: Box::new(kind),
        }
    }

    fn lower_lit(&mut self, expr: &'tcx hir::Expr) -> PatternKind<'tcx> {
        match expr.node {
            hir::ExprLit(ref lit) => {
                let ty = self.tables.expr_ty(expr);
                match lit_to_const(&lit.node, self.tcx, ty, false) {
                    Ok(val) => {
                        let instance = ty::Instance::new(
                            self.tables.local_id_root.expect("literal outside any scope"),
                            self.substs,
                        );
                        let cv = self.tcx.mk_const(ty::Const { val, ty });
                        *self.const_to_pat(instance, cv, expr.hir_id, lit.span).kind
                    },
                    Err(()) => {
                        self.errors.push(PatternError::FloatBug);
                        PatternKind::Wild
                    },
                }
            },
            hir::ExprPath(ref qpath) => *self.lower_path(qpath, expr.hir_id, expr.span).kind,
            hir::ExprUnary(hir::UnNeg, ref expr) => {
                let ty = self.tables.expr_ty(expr);
                let lit = match expr.node {
                    hir::ExprLit(ref lit) => lit,
                    _ => span_bug!(expr.span, "not a literal: {:?}", expr),
                };
                match lit_to_const(&lit.node, self.tcx, ty, true) {
                    Ok(val) => {
                        let instance = ty::Instance::new(
                            self.tables.local_id_root.expect("literal outside any scope"),
                            self.substs,
                        );
                        let cv = self.tcx.mk_const(ty::Const { val, ty });
                        *self.const_to_pat(instance, cv, expr.hir_id, lit.span).kind
                    },
                    Err(()) => {
                        self.errors.push(PatternError::FloatBug);
                        PatternKind::Wild
                    },
                }
            }
            _ => span_bug!(expr.span, "not a literal: {:?}", expr),
        }
    }

    fn const_to_pat(
        &self,
        instance: ty::Instance<'tcx>,
        cv: &'tcx ty::Const<'tcx>,
        id: hir::HirId,
        span: Span,
    ) -> Pattern<'tcx> {
        debug!("const_to_pat: cv={:#?}", cv);
        let kind = match cv.ty.sty {
            ty::TyFloat(_) => {
                let id = self.tcx.hir.hir_to_node_id(id);
                self.tcx.lint_node(
                    ::rustc::lint::builtin::ILLEGAL_FLOATING_POINT_LITERAL_PATTERN,
                    id,
                    span,
                    "floating-point types cannot be used in patterns",
                );
                PatternKind::Constant {
                    value: cv,
                }
            },
            ty::TyAdt(adt_def, _) if adt_def.is_union() => {
                // Matching on union fields is unsafe, we can't hide it in constants
                self.tcx.sess.span_err(span, "cannot use unions in constant patterns");
                PatternKind::Wild
            }
            ty::TyAdt(adt_def, _) if !self.tcx.has_attr(adt_def.did, "structural_match") => {
                let msg = format!("to use a constant of type `{}` in a pattern, \
                                    `{}` must be annotated with `#[derive(PartialEq, Eq)]`",
                                    self.tcx.item_path_str(adt_def.did),
                                    self.tcx.item_path_str(adt_def.did));
                self.tcx.sess.span_err(span, &msg);
                PatternKind::Wild
            },
            ty::TyAdt(adt_def, substs) if adt_def.is_enum() => {
                match cv.val {
                    ConstVal::Value(val) => {
                        let discr = const_discr(
                            self.tcx, self.param_env, instance, val, cv.ty
                        ).unwrap();
                        let variant_index = adt_def
                            .discriminants(self.tcx)
                            .position(|var| var.val == discr)
                            .unwrap();
                        PatternKind::Variant {
                            adt_def,
                            substs,
                            variant_index,
                            subpatterns: adt_def
                                .variants[variant_index]
                                .fields
                                .iter()
                                .enumerate()
                                .map(|(i, _)| {
                                let field = Field::new(i);
                                let val = match cv.val {
                                    ConstVal::Value(miri) => const_val_field(
                                        self.tcx, self.param_env, instance,
                                        Some(variant_index), field, miri, cv.ty,
                                    ).unwrap(),
                                    _ => bug!("{:#?} is not a valid tuple", cv),
                                };
                                FieldPattern {
                                    field,
                                    pattern: self.const_to_pat(instance, val, id, span),
                                }
                            }).collect(),
                        }
                    },
                    _ => return Pattern {
                        span,
                        ty: cv.ty,
                        kind: Box::new(PatternKind::Constant {
                            value: cv,
                        }),
                    }
                }
            },
            ty::TyAdt(adt_def, _) => {
                let struct_var = adt_def.non_enum_variant();
                PatternKind::Leaf {
                    subpatterns: struct_var.fields.iter().enumerate().map(|(i, _)| {
                        let field = Field::new(i);
                        let val = match cv.val {
                            ConstVal::Value(miri) => const_val_field(
                                self.tcx, self.param_env, instance, None, field, miri, cv.ty,
                            ).unwrap(),
                            _ => bug!("{:#?} is not a valid tuple", cv),
                        };
                        FieldPattern {
                            field,
                            pattern: self.const_to_pat(instance, val, id, span),
                        }
                    }).collect()
                }
            }
            ty::TyTuple(fields, _) => {
                PatternKind::Leaf {
                    subpatterns: (0..fields.len()).map(|i| {
                        let field = Field::new(i);
                        let val = match cv.val {
                            ConstVal::Value(miri) => const_val_field(
                                self.tcx, self.param_env, instance, None, field, miri, cv.ty,
                            ).unwrap(),
                            _ => bug!("{:#?} is not a valid tuple", cv),
                        };
                        FieldPattern {
                            field,
                            pattern: self.const_to_pat(instance, val, id, span),
                        }
                    }).collect()
                }
            }
            ty::TyArray(_, n) => {
                PatternKind::Array {
                    prefix: (0..n.val.unwrap_u64()).map(|i| {
                        let i = i as usize;
                        let field = Field::new(i);
                        let val = match cv.val {
                            ConstVal::Value(miri) => const_val_field(
                                self.tcx, self.param_env, instance, None, field, miri, cv.ty,
                            ).unwrap(),
                            _ => bug!("{:#?} is not a valid tuple", cv),
                        };
                        self.const_to_pat(instance, val, id, span)
                    }).collect(),
                    slice: None,
                    suffix: Vec::new(),
                }
            }
            _ => {
                PatternKind::Constant {
                    value: cv,
                }
            },
        };

        Pattern {
            span,
            ty: cv.ty,
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

pub fn compare_const_vals(a: &ConstVal, b: &ConstVal, ty: Ty) -> Option<Ordering> {
    use rustc_const_math::ConstFloat;
    trace!("compare_const_vals: {:?}, {:?}", a, b);
    use rustc::mir::interpret::{Value, PrimVal};
    match (a, b) {
        (&ConstVal::Value(Value::ByVal(PrimVal::Bytes(a))),
         &ConstVal::Value(Value::ByVal(PrimVal::Bytes(b)))) => {
            match ty.sty {
                ty::TyFloat(ty) => {
                    let l = ConstFloat {
                        bits: a,
                        ty,
                    };
                    let r = ConstFloat {
                        bits: b,
                        ty,
                    };
                    // FIXME(oli-obk): report cmp errors?
                    l.try_cmp(r).ok()
                },
                ty::TyInt(_) => Some((a as i128).cmp(&(b as i128))),
                _ => Some(a.cmp(&b)),
            }
        },
        _ if a == b => Some(Ordering::Equal),
        _ => None,
    }
}

fn lit_to_const<'a, 'tcx>(lit: &'tcx ast::LitKind,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          ty: Ty<'tcx>,
                          neg: bool)
                          -> Result<ConstVal<'tcx>, ()> {
    use syntax::ast::*;

    use rustc::mir::interpret::*;
    let lit = match *lit {
        LitKind::Str(ref s, _) => {
            let s = s.as_str();
            let id = tcx.allocate_cached(s.as_bytes());
            let ptr = MemoryPointer::new(id, 0);
            Value::ByValPair(
                PrimVal::Ptr(ptr),
                PrimVal::from_u128(s.len() as u128),
            )
        },
        LitKind::ByteStr(ref data) => {
            let id = tcx.allocate_cached(data);
            let ptr = MemoryPointer::new(id, 0);
            Value::ByVal(PrimVal::Ptr(ptr))
        },
        LitKind::Byte(n) => Value::ByVal(PrimVal::Bytes(n as u128)),
        LitKind::Int(n, _) => {
            enum Int {
                Signed(IntTy),
                Unsigned(UintTy),
            }
            let ty = match ty.sty {
                ty::TyInt(IntTy::Isize) => Int::Signed(tcx.sess.target.isize_ty),
                ty::TyInt(other) => Int::Signed(other),
                ty::TyUint(UintTy::Usize) => Int::Unsigned(tcx.sess.target.usize_ty),
                ty::TyUint(other) => Int::Unsigned(other),
                _ => bug!(),
            };
            let n = match ty {
                // FIXME(oli-obk): are these casts correct?
                Int::Signed(IntTy::I8) if neg =>
                    (n as i128 as i8).overflowing_neg().0 as i128 as u128,
                Int::Signed(IntTy::I16) if neg =>
                    (n as i128 as i16).overflowing_neg().0 as i128 as u128,
                Int::Signed(IntTy::I32) if neg =>
                    (n as i128 as i32).overflowing_neg().0 as i128 as u128,
                Int::Signed(IntTy::I64) if neg =>
                    (n as i128 as i64).overflowing_neg().0 as i128 as u128,
                Int::Signed(IntTy::I128) if neg =>
                    (n as i128).overflowing_neg().0 as u128,
                Int::Signed(IntTy::I8) => n as i128 as i8 as i128 as u128,
                Int::Signed(IntTy::I16) => n as i128 as i16 as i128 as u128,
                Int::Signed(IntTy::I32) => n as i128 as i32 as i128 as u128,
                Int::Signed(IntTy::I64) => n as i128 as i64 as i128 as u128,
                Int::Signed(IntTy::I128) => n,
                Int::Unsigned(UintTy::U8) => n as u8 as u128,
                Int::Unsigned(UintTy::U16) => n as u16 as u128,
                Int::Unsigned(UintTy::U32) => n as u32 as u128,
                Int::Unsigned(UintTy::U64) => n as u64 as u128,
                Int::Unsigned(UintTy::U128) => n,
                _ => bug!(),
            };
            Value::ByVal(PrimVal::Bytes(n))
        },
        LitKind::Float(n, fty) => {
            let n = n.as_str();
            let mut f = parse_float(&n, fty)?;
            if neg {
                f = -f;
            }
            let bits = f.bits;
            Value::ByVal(PrimVal::Bytes(bits))
        }
        LitKind::FloatUnsuffixed(n) => {
            let fty = match ty.sty {
                ty::TyFloat(fty) => fty,
                _ => bug!()
            };
            let n = n.as_str();
            let mut f = parse_float(&n, fty)?;
            if neg {
                f = -f;
            }
            let bits = f.bits;
            Value::ByVal(PrimVal::Bytes(bits))
        }
        LitKind::Bool(b) => Value::ByVal(PrimVal::Bytes(b as u128)),
        LitKind::Char(c) => Value::ByVal(PrimVal::Bytes(c as u128)),
    };
    Ok(ConstVal::Value(lit))
}

fn parse_float<'tcx>(num: &str, fty: ast::FloatTy)
                     -> Result<ConstFloat, ()> {
    ConstFloat::from_str(num, fty).map_err(|_| ())
}
