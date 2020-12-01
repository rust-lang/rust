//! Validation of patterns/matches.

mod check_match;
mod const_to_pat;
mod deconstruct_pat;
mod usefulness;

pub(crate) use self::check_match::check_match;

use crate::thir::util::UserAnnotatedTyHelpers;

use rustc_ast as ast;
use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, CtorOf, DefKind, Res};
use rustc_hir::pat_util::EnumerateAndAdjustIterator;
use rustc_hir::RangeEnd;
use rustc_index::vec::Idx;
use rustc_middle::mir::interpret::{get_slice_bytes, ConstValue};
use rustc_middle::mir::interpret::{ErrorHandled, LitToConstError, LitToConstInput};
use rustc_middle::mir::UserTypeProjection;
use rustc_middle::mir::{BorrowKind, Field, Mutability};
use rustc_middle::ty::subst::{GenericArg, SubstsRef};
use rustc_middle::ty::{self, AdtDef, DefIdTree, Region, Ty, TyCtxt, UserType};
use rustc_middle::ty::{
    CanonicalUserType, CanonicalUserTypeAnnotation, CanonicalUserTypeAnnotations,
};
use rustc_span::{Span, Symbol, DUMMY_SP};
use rustc_target::abi::VariantIdx;

use std::cmp::Ordering;
use std::fmt;

#[derive(Clone, Debug)]
crate enum PatternError {
    AssocConstInPattern(Span),
    ConstParamInPattern(Span),
    StaticInPattern(Span),
    FloatBug,
    NonConstPath(Span),
}

#[derive(Copy, Clone, Debug, PartialEq)]
crate enum BindingMode {
    ByValue,
    ByRef(BorrowKind),
}

#[derive(Clone, Debug, PartialEq)]
crate struct FieldPat<'tcx> {
    crate field: Field,
    crate pattern: Pat<'tcx>,
}

#[derive(Clone, Debug, PartialEq)]
crate struct Pat<'tcx> {
    crate ty: Ty<'tcx>,
    crate span: Span,
    crate kind: Box<PatKind<'tcx>>,
}

impl<'tcx> Pat<'tcx> {
    pub(crate) fn wildcard_from_ty(ty: Ty<'tcx>) -> Self {
        Pat { ty, span: DUMMY_SP, kind: Box::new(PatKind::Wild) }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
crate struct PatTyProj<'tcx> {
    crate user_ty: CanonicalUserType<'tcx>,
}

impl<'tcx> PatTyProj<'tcx> {
    pub(crate) fn from_user_type(user_annotation: CanonicalUserType<'tcx>) -> Self {
        Self { user_ty: user_annotation }
    }

    pub(crate) fn user_ty(
        self,
        annotations: &mut CanonicalUserTypeAnnotations<'tcx>,
        inferred_ty: Ty<'tcx>,
        span: Span,
    ) -> UserTypeProjection {
        UserTypeProjection {
            base: annotations.push(CanonicalUserTypeAnnotation {
                span,
                user_ty: self.user_ty,
                inferred_ty,
            }),
            projs: Vec::new(),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
crate struct Ascription<'tcx> {
    crate user_ty: PatTyProj<'tcx>,
    /// Variance to use when relating the type `user_ty` to the **type of the value being
    /// matched**. Typically, this is `Variance::Covariant`, since the value being matched must
    /// have a type that is some subtype of the ascribed type.
    ///
    /// Note that this variance does not apply for any bindings within subpatterns. The type
    /// assigned to those bindings must be exactly equal to the `user_ty` given here.
    ///
    /// The only place where this field is not `Covariant` is when matching constants, where
    /// we currently use `Contravariant` -- this is because the constant type just needs to
    /// be "comparable" to the type of the input value. So, for example:
    ///
    /// ```text
    /// match x { "foo" => .. }
    /// ```
    ///
    /// requires that `&'static str <: T_x`, where `T_x` is the type of `x`. Really, we should
    /// probably be checking for a `PartialEq` impl instead, but this preserves the behavior
    /// of the old type-check for now. See #57280 for details.
    crate variance: ty::Variance,
    crate user_ty_span: Span,
}

#[derive(Clone, Debug, PartialEq)]
crate enum PatKind<'tcx> {
    Wild,

    AscribeUserType {
        ascription: Ascription<'tcx>,
        subpattern: Pat<'tcx>,
    },

    /// `x`, `ref x`, `x @ P`, etc.
    Binding {
        mutability: Mutability,
        name: Symbol,
        mode: BindingMode,
        var: hir::HirId,
        ty: Ty<'tcx>,
        subpattern: Option<Pat<'tcx>>,
        /// Is this the leftmost occurrence of the binding, i.e., is `var` the
        /// `HirId` of this pattern?
        is_primary: bool,
    },

    /// `Foo(...)` or `Foo{...}` or `Foo`, where `Foo` is a variant name from an ADT with
    /// multiple variants.
    Variant {
        adt_def: &'tcx AdtDef,
        substs: SubstsRef<'tcx>,
        variant_index: VariantIdx,
        subpatterns: Vec<FieldPat<'tcx>>,
    },

    /// `(...)`, `Foo(...)`, `Foo{...}`, or `Foo`, where `Foo` is a variant name from an ADT with
    /// a single variant.
    Leaf {
        subpatterns: Vec<FieldPat<'tcx>>,
    },

    /// `box P`, `&P`, `&mut P`, etc.
    Deref {
        subpattern: Pat<'tcx>,
    },

    /// One of the following:
    /// * `&str`, which will be handled as a string pattern and thus exhaustiveness
    ///   checking will detect if you use the same string twice in different patterns.
    /// * integer, bool, char or float, which will be handled by exhaustivenes to cover exactly
    ///   its own value, similar to `&str`, but these values are much simpler.
    /// * Opaque constants, that must not be matched structurally. So anything that does not derive
    ///   `PartialEq` and `Eq`.
    Constant {
        value: &'tcx ty::Const<'tcx>,
    },

    Range(PatRange<'tcx>),

    /// Matches against a slice, checking the length and extracting elements.
    /// irrefutable when there is a slice pattern and both `prefix` and `suffix` are empty.
    /// e.g., `&[ref xs @ ..]`.
    Slice {
        prefix: Vec<Pat<'tcx>>,
        slice: Option<Pat<'tcx>>,
        suffix: Vec<Pat<'tcx>>,
    },

    /// Fixed match against an array; irrefutable.
    Array {
        prefix: Vec<Pat<'tcx>>,
        slice: Option<Pat<'tcx>>,
        suffix: Vec<Pat<'tcx>>,
    },

    /// An or-pattern, e.g. `p | q`.
    /// Invariant: `pats.len() >= 2`.
    Or {
        pats: Vec<Pat<'tcx>>,
    },
}

#[derive(Copy, Clone, Debug, PartialEq)]
crate struct PatRange<'tcx> {
    crate lo: &'tcx ty::Const<'tcx>,
    crate hi: &'tcx ty::Const<'tcx>,
    crate end: RangeEnd,
}

impl<'tcx> fmt::Display for Pat<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Printing lists is a chore.
        let mut first = true;
        let mut start_or_continue = |s| {
            if first {
                first = false;
                ""
            } else {
                s
            }
        };
        let mut start_or_comma = || start_or_continue(", ");

        match *self.kind {
            PatKind::Wild => write!(f, "_"),
            PatKind::AscribeUserType { ref subpattern, .. } => write!(f, "{}: _", subpattern),
            PatKind::Binding { mutability, name, mode, ref subpattern, .. } => {
                let is_mut = match mode {
                    BindingMode::ByValue => mutability == Mutability::Mut,
                    BindingMode::ByRef(bk) => {
                        write!(f, "ref ")?;
                        matches!(bk, BorrowKind::Mut { .. })
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
            PatKind::Variant { ref subpatterns, .. } | PatKind::Leaf { ref subpatterns } => {
                let variant = match *self.kind {
                    PatKind::Variant { adt_def, variant_index, .. } => {
                        Some(&adt_def.variants[variant_index])
                    }
                    _ => {
                        if let ty::Adt(adt, _) = self.ty.kind() {
                            if !adt.is_enum() {
                                Some(&adt.variants[VariantIdx::new(0)])
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                };

                if let Some(variant) = variant {
                    write!(f, "{}", variant.ident)?;

                    // Only for Adt we can have `S {...}`,
                    // which we handle separately here.
                    if variant.ctor_kind == CtorKind::Fictive {
                        write!(f, " {{ ")?;

                        let mut printed = 0;
                        for p in subpatterns {
                            if let PatKind::Wild = *p.pattern.kind {
                                continue;
                            }
                            let name = variant.fields[p.field.index()].ident;
                            write!(f, "{}{}: {}", start_or_comma(), name, p.pattern)?;
                            printed += 1;
                        }

                        if printed < variant.fields.len() {
                            write!(f, "{}..", start_or_comma())?;
                        }

                        return write!(f, " }}");
                    }
                }

                let num_fields = variant.map_or(subpatterns.len(), |v| v.fields.len());
                if num_fields != 0 || variant.is_none() {
                    write!(f, "(")?;
                    for i in 0..num_fields {
                        write!(f, "{}", start_or_comma())?;

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
            PatKind::Deref { ref subpattern } => {
                match self.ty.kind() {
                    ty::Adt(def, _) if def.is_box() => write!(f, "box ")?,
                    ty::Ref(_, _, mutbl) => {
                        write!(f, "&{}", mutbl.prefix_str())?;
                    }
                    _ => bug!("{} is a bad Deref pattern type", self.ty),
                }
                write!(f, "{}", subpattern)
            }
            PatKind::Constant { value } => write!(f, "{}", value),
            PatKind::Range(PatRange { lo, hi, end }) => {
                write!(f, "{}", lo)?;
                write!(f, "{}", end)?;
                write!(f, "{}", hi)
            }
            PatKind::Slice { ref prefix, ref slice, ref suffix }
            | PatKind::Array { ref prefix, ref slice, ref suffix } => {
                write!(f, "[")?;
                for p in prefix {
                    write!(f, "{}{}", start_or_comma(), p)?;
                }
                if let Some(ref slice) = *slice {
                    write!(f, "{}", start_or_comma())?;
                    match *slice.kind {
                        PatKind::Wild => {}
                        _ => write!(f, "{}", slice)?,
                    }
                    write!(f, "..")?;
                }
                for p in suffix {
                    write!(f, "{}{}", start_or_comma(), p)?;
                }
                write!(f, "]")
            }
            PatKind::Or { ref pats } => {
                for pat in pats {
                    write!(f, "{}{}", start_or_continue(" | "), pat)?;
                }
                Ok(())
            }
        }
    }
}

crate struct PatCtxt<'a, 'tcx> {
    crate tcx: TyCtxt<'tcx>,
    crate param_env: ty::ParamEnv<'tcx>,
    crate typeck_results: &'a ty::TypeckResults<'tcx>,
    crate errors: Vec<PatternError>,
    include_lint_checks: bool,
}

impl<'a, 'tcx> Pat<'tcx> {
    crate fn from_hir(
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        typeck_results: &'a ty::TypeckResults<'tcx>,
        pat: &'tcx hir::Pat<'tcx>,
    ) -> Self {
        let mut pcx = PatCtxt::new(tcx, param_env, typeck_results);
        let result = pcx.lower_pattern(pat);
        if !pcx.errors.is_empty() {
            let msg = format!("encountered errors lowering pattern: {:?}", pcx.errors);
            tcx.sess.delay_span_bug(pat.span, &msg);
        }
        debug!("Pat::from_hir({:?}) = {:?}", pat, result);
        result
    }
}

impl<'a, 'tcx> PatCtxt<'a, 'tcx> {
    crate fn new(
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        typeck_results: &'a ty::TypeckResults<'tcx>,
    ) -> Self {
        PatCtxt { tcx, param_env, typeck_results, errors: vec![], include_lint_checks: false }
    }

    crate fn include_lint_checks(&mut self) -> &mut Self {
        self.include_lint_checks = true;
        self
    }

    crate fn lower_pattern(&mut self, pat: &'tcx hir::Pat<'tcx>) -> Pat<'tcx> {
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
        // Applying the adjustments, we want to instead output `&&Some(n)` (as a THIR pattern). So
        // we wrap the unadjusted pattern in `PatKind::Deref` repeatedly, consuming the
        // adjustments in *reverse order* (last-in-first-out, so that the last `Deref` inserted
        // gets the least-dereferenced type).
        let unadjusted_pat = self.lower_pattern_unadjusted(pat);
        self.typeck_results.pat_adjustments().get(pat.hir_id).unwrap_or(&vec![]).iter().rev().fold(
            unadjusted_pat,
            |pat, ref_ty| {
                debug!("{:?}: wrapping pattern with type {:?}", pat, ref_ty);
                Pat {
                    span: pat.span,
                    ty: ref_ty,
                    kind: Box::new(PatKind::Deref { subpattern: pat }),
                }
            },
        )
    }

    fn lower_range_expr(
        &mut self,
        expr: &'tcx hir::Expr<'tcx>,
    ) -> (PatKind<'tcx>, Option<Ascription<'tcx>>) {
        match self.lower_lit(expr) {
            PatKind::AscribeUserType { ascription, subpattern: Pat { kind: box kind, .. } } => {
                (kind, Some(ascription))
            }
            kind => (kind, None),
        }
    }

    fn lower_pattern_range(
        &mut self,
        ty: Ty<'tcx>,
        lo: &'tcx ty::Const<'tcx>,
        hi: &'tcx ty::Const<'tcx>,
        end: RangeEnd,
        span: Span,
    ) -> PatKind<'tcx> {
        assert_eq!(lo.ty, ty);
        assert_eq!(hi.ty, ty);
        let cmp = compare_const_vals(self.tcx, lo, hi, self.param_env, ty);
        match (end, cmp) {
            // `x..y` where `x < y`.
            // Non-empty because the range includes at least `x`.
            (RangeEnd::Excluded, Some(Ordering::Less)) => PatKind::Range(PatRange { lo, hi, end }),
            // `x..y` where `x >= y`. The range is empty => error.
            (RangeEnd::Excluded, _) => {
                struct_span_err!(
                    self.tcx.sess,
                    span,
                    E0579,
                    "lower range bound must be less than upper"
                )
                .emit();
                PatKind::Wild
            }
            // `x..=y` where `x == y`.
            (RangeEnd::Included, Some(Ordering::Equal)) => PatKind::Constant { value: lo },
            // `x..=y` where `x < y`.
            (RangeEnd::Included, Some(Ordering::Less)) => PatKind::Range(PatRange { lo, hi, end }),
            // `x..=y` where `x > y` hence the range is empty => error.
            (RangeEnd::Included, _) => {
                let mut err = struct_span_err!(
                    self.tcx.sess,
                    span,
                    E0030,
                    "lower range bound must be less than or equal to upper"
                );
                err.span_label(span, "lower bound larger than upper bound");
                if self.tcx.sess.teach(&err.get_code().unwrap()) {
                    err.note(
                        "When matching against a range, the compiler \
                              verifies that the range is non-empty. Range \
                              patterns include both end-points, so this is \
                              equivalent to requiring the start of the range \
                              to be less than or equal to the end of the range.",
                    );
                }
                err.emit();
                PatKind::Wild
            }
        }
    }

    fn normalize_range_pattern_ends(
        &self,
        ty: Ty<'tcx>,
        lo: Option<&PatKind<'tcx>>,
        hi: Option<&PatKind<'tcx>>,
    ) -> Option<(&'tcx ty::Const<'tcx>, &'tcx ty::Const<'tcx>)> {
        match (lo, hi) {
            (Some(PatKind::Constant { value: lo }), Some(PatKind::Constant { value: hi })) => {
                Some((lo, hi))
            }
            (Some(PatKind::Constant { value: lo }), None) => {
                Some((lo, ty.numeric_max_val(self.tcx)?))
            }
            (None, Some(PatKind::Constant { value: hi })) => {
                Some((ty.numeric_min_val(self.tcx)?, hi))
            }
            _ => None,
        }
    }

    fn lower_pattern_unadjusted(&mut self, pat: &'tcx hir::Pat<'tcx>) -> Pat<'tcx> {
        let mut ty = self.typeck_results.node_type(pat.hir_id);

        let kind = match pat.kind {
            hir::PatKind::Wild => PatKind::Wild,

            hir::PatKind::Lit(ref value) => self.lower_lit(value),

            hir::PatKind::Range(ref lo_expr, ref hi_expr, end) => {
                let (lo_expr, hi_expr) = (lo_expr.as_deref(), hi_expr.as_deref());
                let lo_span = lo_expr.map_or(pat.span, |e| e.span);
                let lo = lo_expr.map(|e| self.lower_range_expr(e));
                let hi = hi_expr.map(|e| self.lower_range_expr(e));

                let (lp, hp) = (lo.as_ref().map(|x| &x.0), hi.as_ref().map(|x| &x.0));
                let mut kind = match self.normalize_range_pattern_ends(ty, lp, hp) {
                    Some((lc, hc)) => self.lower_pattern_range(ty, lc, hc, end, lo_span),
                    None => {
                        let msg = &format!(
                            "found bad range pattern `{:?}` outside of error recovery",
                            (&lo, &hi),
                        );
                        self.tcx.sess.delay_span_bug(pat.span, msg);
                        PatKind::Wild
                    }
                };

                // If we are handling a range with associated constants (e.g.
                // `Foo::<'a>::A..=Foo::B`), we need to put the ascriptions for the associated
                // constants somewhere. Have them on the range pattern.
                for end in &[lo, hi] {
                    if let Some((_, Some(ascription))) = end {
                        let subpattern = Pat { span: pat.span, ty, kind: Box::new(kind) };
                        kind = PatKind::AscribeUserType { ascription: *ascription, subpattern };
                    }
                }

                kind
            }

            hir::PatKind::Path(ref qpath) => {
                return self.lower_path(qpath, pat.hir_id, pat.span);
            }

            hir::PatKind::Ref(ref subpattern, _) | hir::PatKind::Box(ref subpattern) => {
                PatKind::Deref { subpattern: self.lower_pattern(subpattern) }
            }

            hir::PatKind::Slice(ref prefix, ref slice, ref suffix) => {
                self.slice_or_array_pattern(pat.span, ty, prefix, slice, suffix)
            }

            hir::PatKind::Tuple(ref pats, ddpos) => {
                let tys = match ty.kind() {
                    ty::Tuple(ref tys) => tys,
                    _ => span_bug!(pat.span, "unexpected type for tuple pattern: {:?}", ty),
                };
                let subpatterns = self.lower_tuple_subpats(pats, tys.len(), ddpos);
                PatKind::Leaf { subpatterns }
            }

            hir::PatKind::Binding(_, id, ident, ref sub) => {
                let bm = *self
                    .typeck_results
                    .pat_binding_modes()
                    .get(pat.hir_id)
                    .expect("missing binding mode");
                let (mutability, mode) = match bm {
                    ty::BindByValue(mutbl) => (mutbl, BindingMode::ByValue),
                    ty::BindByReference(hir::Mutability::Mut) => (
                        Mutability::Not,
                        BindingMode::ByRef(BorrowKind::Mut { allow_two_phase_borrow: false }),
                    ),
                    ty::BindByReference(hir::Mutability::Not) => {
                        (Mutability::Not, BindingMode::ByRef(BorrowKind::Shared))
                    }
                };

                // A ref x pattern is the same node used for x, and as such it has
                // x's type, which is &T, where we want T (the type being matched).
                let var_ty = ty;
                if let ty::BindByReference(_) = bm {
                    if let ty::Ref(_, rty, _) = ty.kind() {
                        ty = rty;
                    } else {
                        bug!("`ref {}` has wrong type {}", ident, ty);
                    }
                };

                PatKind::Binding {
                    mutability,
                    mode,
                    name: ident.name,
                    var: id,
                    ty: var_ty,
                    subpattern: self.lower_opt_pattern(sub),
                    is_primary: id == pat.hir_id,
                }
            }

            hir::PatKind::TupleStruct(ref qpath, ref pats, ddpos) => {
                let res = self.typeck_results.qpath_res(qpath, pat.hir_id);
                let adt_def = match ty.kind() {
                    ty::Adt(adt_def, _) => adt_def,
                    _ => span_bug!(pat.span, "tuple struct pattern not applied to an ADT {:?}", ty),
                };
                let variant_def = adt_def.variant_of_res(res);
                let subpatterns = self.lower_tuple_subpats(pats, variant_def.fields.len(), ddpos);
                self.lower_variant_or_leaf(res, pat.hir_id, pat.span, ty, subpatterns)
            }

            hir::PatKind::Struct(ref qpath, ref fields, _) => {
                let res = self.typeck_results.qpath_res(qpath, pat.hir_id);
                let subpatterns = fields
                    .iter()
                    .map(|field| FieldPat {
                        field: Field::new(self.tcx.field_index(field.hir_id, self.typeck_results)),
                        pattern: self.lower_pattern(&field.pat),
                    })
                    .collect();

                self.lower_variant_or_leaf(res, pat.hir_id, pat.span, ty, subpatterns)
            }

            hir::PatKind::Or(ref pats) => PatKind::Or { pats: self.lower_patterns(pats) },
        };

        Pat { span: pat.span, ty, kind: Box::new(kind) }
    }

    fn lower_tuple_subpats(
        &mut self,
        pats: &'tcx [&'tcx hir::Pat<'tcx>],
        expected_len: usize,
        gap_pos: Option<usize>,
    ) -> Vec<FieldPat<'tcx>> {
        pats.iter()
            .enumerate_and_adjust(expected_len, gap_pos)
            .map(|(i, subpattern)| FieldPat {
                field: Field::new(i),
                pattern: self.lower_pattern(subpattern),
            })
            .collect()
    }

    fn lower_patterns(&mut self, pats: &'tcx [&'tcx hir::Pat<'tcx>]) -> Vec<Pat<'tcx>> {
        pats.iter().map(|p| self.lower_pattern(p)).collect()
    }

    fn lower_opt_pattern(&mut self, pat: &'tcx Option<&'tcx hir::Pat<'tcx>>) -> Option<Pat<'tcx>> {
        pat.as_ref().map(|p| self.lower_pattern(p))
    }

    fn slice_or_array_pattern(
        &mut self,
        span: Span,
        ty: Ty<'tcx>,
        prefix: &'tcx [&'tcx hir::Pat<'tcx>],
        slice: &'tcx Option<&'tcx hir::Pat<'tcx>>,
        suffix: &'tcx [&'tcx hir::Pat<'tcx>],
    ) -> PatKind<'tcx> {
        let prefix = self.lower_patterns(prefix);
        let slice = self.lower_opt_pattern(slice);
        let suffix = self.lower_patterns(suffix);
        match ty.kind() {
            // Matching a slice, `[T]`.
            ty::Slice(..) => PatKind::Slice { prefix, slice, suffix },
            // Fixed-length array, `[T; len]`.
            ty::Array(_, len) => {
                let len = len.eval_usize(self.tcx, self.param_env);
                assert!(len >= prefix.len() as u64 + suffix.len() as u64);
                PatKind::Array { prefix, slice, suffix }
            }
            _ => span_bug!(span, "bad slice pattern type {:?}", ty),
        }
    }

    fn lower_variant_or_leaf(
        &mut self,
        res: Res,
        hir_id: hir::HirId,
        span: Span,
        ty: Ty<'tcx>,
        subpatterns: Vec<FieldPat<'tcx>>,
    ) -> PatKind<'tcx> {
        let res = match res {
            Res::Def(DefKind::Ctor(CtorOf::Variant, ..), variant_ctor_id) => {
                let variant_id = self.tcx.parent(variant_ctor_id).unwrap();
                Res::Def(DefKind::Variant, variant_id)
            }
            res => res,
        };

        let mut kind = match res {
            Res::Def(DefKind::Variant, variant_id) => {
                let enum_id = self.tcx.parent(variant_id).unwrap();
                let adt_def = self.tcx.adt_def(enum_id);
                if adt_def.is_enum() {
                    let substs = match ty.kind() {
                        ty::Adt(_, substs) | ty::FnDef(_, substs) => substs,
                        ty::Error(_) => {
                            // Avoid ICE (#50585)
                            return PatKind::Wild;
                        }
                        _ => bug!("inappropriate type for def: {:?}", ty),
                    };
                    PatKind::Variant {
                        adt_def,
                        substs,
                        variant_index: adt_def.variant_index_with_id(variant_id),
                        subpatterns,
                    }
                } else {
                    PatKind::Leaf { subpatterns }
                }
            }

            Res::Def(
                DefKind::Struct
                | DefKind::Ctor(CtorOf::Struct, ..)
                | DefKind::Union
                | DefKind::TyAlias
                | DefKind::AssocTy,
                _,
            )
            | Res::SelfTy(..)
            | Res::SelfCtor(..) => PatKind::Leaf { subpatterns },
            _ => {
                let pattern_error = match res {
                    Res::Def(DefKind::ConstParam, _) => PatternError::ConstParamInPattern(span),
                    _ => PatternError::NonConstPath(span),
                };
                self.errors.push(pattern_error);
                PatKind::Wild
            }
        };

        if let Some(user_ty) = self.user_substs_applied_to_ty_of_hir_id(hir_id) {
            debug!("lower_variant_or_leaf: kind={:?} user_ty={:?} span={:?}", kind, user_ty, span);
            kind = PatKind::AscribeUserType {
                subpattern: Pat { span, ty, kind: Box::new(kind) },
                ascription: Ascription {
                    user_ty: PatTyProj::from_user_type(user_ty),
                    user_ty_span: span,
                    variance: ty::Variance::Covariant,
                },
            };
        }

        kind
    }

    /// Takes a HIR Path. If the path is a constant, evaluates it and feeds
    /// it to `const_to_pat`. Any other path (like enum variants without fields)
    /// is converted to the corresponding pattern via `lower_variant_or_leaf`.
    fn lower_path(&mut self, qpath: &hir::QPath<'_>, id: hir::HirId, span: Span) -> Pat<'tcx> {
        let ty = self.typeck_results.node_type(id);
        let res = self.typeck_results.qpath_res(qpath, id);

        let pat_from_kind = |kind| Pat { span, ty, kind: Box::new(kind) };

        let (def_id, is_associated_const) = match res {
            Res::Def(DefKind::Const, def_id) => (def_id, false),
            Res::Def(DefKind::AssocConst, def_id) => (def_id, true),

            _ => return pat_from_kind(self.lower_variant_or_leaf(res, id, span, ty, vec![])),
        };

        // Use `Reveal::All` here because patterns are always monomorphic even if their function
        // isn't.
        let param_env_reveal_all = self.param_env.with_reveal_all_normalized(self.tcx);
        let substs = self.typeck_results.node_substs(id);
        let instance = match ty::Instance::resolve(self.tcx, param_env_reveal_all, def_id, substs) {
            Ok(Some(i)) => i,
            Ok(None) => {
                self.errors.push(if is_associated_const {
                    PatternError::AssocConstInPattern(span)
                } else {
                    PatternError::StaticInPattern(span)
                });

                return pat_from_kind(PatKind::Wild);
            }

            Err(_) => {
                self.tcx.sess.span_err(span, "could not evaluate constant pattern");
                return pat_from_kind(PatKind::Wild);
            }
        };

        // `mir_const_qualif` must be called with the `DefId` of the item where the const is
        // defined, not where it is declared. The difference is significant for associated
        // constants.
        let mir_structural_match_violation = self.tcx.mir_const_qualif(instance.def_id()).custom_eq;
        debug!("mir_structural_match_violation({:?}) -> {}", qpath, mir_structural_match_violation);

        match self.tcx.const_eval_instance(param_env_reveal_all, instance, Some(span)) {
            Ok(value) => {
                let const_ =
                    ty::Const::from_value(self.tcx, value, self.typeck_results.node_type(id));

                let pattern = self.const_to_pat(&const_, id, span, mir_structural_match_violation);

                if !is_associated_const {
                    return pattern;
                }

                let user_provided_types = self.typeck_results().user_provided_types();
                if let Some(u_ty) = user_provided_types.get(id) {
                    let user_ty = PatTyProj::from_user_type(*u_ty);
                    Pat {
                        span,
                        kind: Box::new(PatKind::AscribeUserType {
                            subpattern: pattern,
                            ascription: Ascription {
                                /// Note that use `Contravariant` here. See the
                                /// `variance` field documentation for details.
                                variance: ty::Variance::Contravariant,
                                user_ty,
                                user_ty_span: span,
                            },
                        }),
                        ty: const_.ty,
                    }
                } else {
                    pattern
                }
            }
            Err(ErrorHandled::TooGeneric) => {
                // While `Reported | Linted` cases will have diagnostics emitted already
                // it is not true for TooGeneric case, so we need to give user more information.
                self.tcx.sess.span_err(span, "constant pattern depends on a generic parameter");
                pat_from_kind(PatKind::Wild)
            }
            Err(_) => {
                self.tcx.sess.span_err(span, "could not evaluate constant pattern");
                pat_from_kind(PatKind::Wild)
            }
        }
    }

    /// Converts literals, paths and negation of literals to patterns.
    /// The special case for negation exists to allow things like `-128_i8`
    /// which would overflow if we tried to evaluate `128_i8` and then negate
    /// afterwards.
    fn lower_lit(&mut self, expr: &'tcx hir::Expr<'tcx>) -> PatKind<'tcx> {
        if let hir::ExprKind::Path(ref qpath) = expr.kind {
            *self.lower_path(qpath, expr.hir_id, expr.span).kind
        } else {
            let (lit, neg) = match expr.kind {
                hir::ExprKind::ConstBlock(ref anon_const) => {
                    let anon_const_def_id = self.tcx.hir().local_def_id(anon_const.hir_id);
                    let value = ty::Const::from_anon_const(self.tcx, anon_const_def_id);
                    return *self.const_to_pat(value, expr.hir_id, expr.span, false).kind;
                }
                hir::ExprKind::Lit(ref lit) => (lit, false),
                hir::ExprKind::Unary(hir::UnOp::UnNeg, ref expr) => {
                    let lit = match expr.kind {
                        hir::ExprKind::Lit(ref lit) => lit,
                        _ => span_bug!(expr.span, "not a literal: {:?}", expr),
                    };
                    (lit, true)
                }
                _ => span_bug!(expr.span, "not a literal: {:?}", expr),
            };

            let lit_input =
                LitToConstInput { lit: &lit.node, ty: self.typeck_results.expr_ty(expr), neg };
            match self.tcx.at(expr.span).lit_to_const(lit_input) {
                Ok(val) => *self.const_to_pat(val, expr.hir_id, lit.span, false).kind,
                Err(LitToConstError::UnparseableFloat) => {
                    self.errors.push(PatternError::FloatBug);
                    PatKind::Wild
                }
                Err(LitToConstError::Reported) => PatKind::Wild,
                Err(LitToConstError::TypeError) => bug!("lower_lit: had type error"),
            }
        }
    }
}

impl<'tcx> UserAnnotatedTyHelpers<'tcx> for PatCtxt<'_, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn typeck_results(&self) -> &ty::TypeckResults<'tcx> {
        self.typeck_results
    }
}

crate trait PatternFoldable<'tcx>: Sized {
    fn fold_with<F: PatternFolder<'tcx>>(&self, folder: &mut F) -> Self {
        self.super_fold_with(folder)
    }

    fn super_fold_with<F: PatternFolder<'tcx>>(&self, folder: &mut F) -> Self;
}

crate trait PatternFolder<'tcx>: Sized {
    fn fold_pattern(&mut self, pattern: &Pat<'tcx>) -> Pat<'tcx> {
        pattern.super_fold_with(self)
    }

    fn fold_pattern_kind(&mut self, kind: &PatKind<'tcx>) -> PatKind<'tcx> {
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
    fn super_fold_with<F: PatternFolder<'tcx>>(&self, folder: &mut F) -> Self {
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

CloneImpls! { <'tcx>
    Span, Field, Mutability, Symbol, hir::HirId, usize, ty::Const<'tcx>,
    Region<'tcx>, Ty<'tcx>, BindingMode, &'tcx AdtDef,
    SubstsRef<'tcx>, &'tcx GenericArg<'tcx>, UserType<'tcx>,
    UserTypeProjection, PatTyProj<'tcx>
}

impl<'tcx> PatternFoldable<'tcx> for FieldPat<'tcx> {
    fn super_fold_with<F: PatternFolder<'tcx>>(&self, folder: &mut F) -> Self {
        FieldPat { field: self.field.fold_with(folder), pattern: self.pattern.fold_with(folder) }
    }
}

impl<'tcx> PatternFoldable<'tcx> for Pat<'tcx> {
    fn fold_with<F: PatternFolder<'tcx>>(&self, folder: &mut F) -> Self {
        folder.fold_pattern(self)
    }

    fn super_fold_with<F: PatternFolder<'tcx>>(&self, folder: &mut F) -> Self {
        Pat {
            ty: self.ty.fold_with(folder),
            span: self.span.fold_with(folder),
            kind: self.kind.fold_with(folder),
        }
    }
}

impl<'tcx> PatternFoldable<'tcx> for PatKind<'tcx> {
    fn fold_with<F: PatternFolder<'tcx>>(&self, folder: &mut F) -> Self {
        folder.fold_pattern_kind(self)
    }

    fn super_fold_with<F: PatternFolder<'tcx>>(&self, folder: &mut F) -> Self {
        match *self {
            PatKind::Wild => PatKind::Wild,
            PatKind::AscribeUserType {
                ref subpattern,
                ascription: Ascription { variance, ref user_ty, user_ty_span },
            } => PatKind::AscribeUserType {
                subpattern: subpattern.fold_with(folder),
                ascription: Ascription {
                    user_ty: user_ty.fold_with(folder),
                    variance,
                    user_ty_span,
                },
            },
            PatKind::Binding { mutability, name, mode, var, ty, ref subpattern, is_primary } => {
                PatKind::Binding {
                    mutability: mutability.fold_with(folder),
                    name: name.fold_with(folder),
                    mode: mode.fold_with(folder),
                    var: var.fold_with(folder),
                    ty: ty.fold_with(folder),
                    subpattern: subpattern.fold_with(folder),
                    is_primary,
                }
            }
            PatKind::Variant { adt_def, substs, variant_index, ref subpatterns } => {
                PatKind::Variant {
                    adt_def: adt_def.fold_with(folder),
                    substs: substs.fold_with(folder),
                    variant_index,
                    subpatterns: subpatterns.fold_with(folder),
                }
            }
            PatKind::Leaf { ref subpatterns } => {
                PatKind::Leaf { subpatterns: subpatterns.fold_with(folder) }
            }
            PatKind::Deref { ref subpattern } => {
                PatKind::Deref { subpattern: subpattern.fold_with(folder) }
            }
            PatKind::Constant { value } => PatKind::Constant { value },
            PatKind::Range(range) => PatKind::Range(range),
            PatKind::Slice { ref prefix, ref slice, ref suffix } => PatKind::Slice {
                prefix: prefix.fold_with(folder),
                slice: slice.fold_with(folder),
                suffix: suffix.fold_with(folder),
            },
            PatKind::Array { ref prefix, ref slice, ref suffix } => PatKind::Array {
                prefix: prefix.fold_with(folder),
                slice: slice.fold_with(folder),
                suffix: suffix.fold_with(folder),
            },
            PatKind::Or { ref pats } => PatKind::Or { pats: pats.fold_with(folder) },
        }
    }
}

crate fn compare_const_vals<'tcx>(
    tcx: TyCtxt<'tcx>,
    a: &'tcx ty::Const<'tcx>,
    b: &'tcx ty::Const<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    ty: Ty<'tcx>,
) -> Option<Ordering> {
    trace!("compare_const_vals: {:?}, {:?}", a, b);

    let from_bool = |v: bool| v.then_some(Ordering::Equal);

    let fallback = || from_bool(a == b);

    // Use the fallback if any type differs
    if a.ty != b.ty || a.ty != ty {
        return fallback();
    }

    // Early return for equal constants (so e.g. references to ZSTs can be compared, even if they
    // are just integer addresses).
    if a.val == b.val {
        return from_bool(true);
    }

    let a_bits = a.try_eval_bits(tcx, param_env, ty);
    let b_bits = b.try_eval_bits(tcx, param_env, ty);

    if let (Some(a), Some(b)) = (a_bits, b_bits) {
        use rustc_apfloat::Float;
        return match *ty.kind() {
            ty::Float(ast::FloatTy::F32) => {
                let l = rustc_apfloat::ieee::Single::from_bits(a);
                let r = rustc_apfloat::ieee::Single::from_bits(b);
                l.partial_cmp(&r)
            }
            ty::Float(ast::FloatTy::F64) => {
                let l = rustc_apfloat::ieee::Double::from_bits(a);
                let r = rustc_apfloat::ieee::Double::from_bits(b);
                l.partial_cmp(&r)
            }
            ty::Int(ity) => {
                use rustc_attr::SignedInt;
                use rustc_middle::ty::layout::IntegerExt;
                let size = rustc_target::abi::Integer::from_attr(&tcx, SignedInt(ity)).size();
                let a = size.sign_extend(a);
                let b = size.sign_extend(b);
                Some((a as i128).cmp(&(b as i128)))
            }
            _ => Some(a.cmp(&b)),
        };
    }

    if let ty::Str = ty.kind() {
        if let (
            ty::ConstKind::Value(a_val @ ConstValue::Slice { .. }),
            ty::ConstKind::Value(b_val @ ConstValue::Slice { .. }),
        ) = (a.val, b.val)
        {
            let a_bytes = get_slice_bytes(&tcx, a_val);
            let b_bytes = get_slice_bytes(&tcx, b_val);
            return from_bool(a_bytes == b_bytes);
        }
    }
    fallback()
}
