//! Validation of matches.
//!
//! This module provides lowering from [hir_def::expr::Pat] to [self::Pat] and match
//! checking algorithm.
//!
//! It is a loose port of `rustc_mir_build::thir::pattern` module.

mod deconstruct_pat;
mod pat_util;
pub(crate) mod usefulness;

use hir_def::{body::Body, EnumVariantId, LocalFieldId, VariantId};
use la_arena::Idx;

use crate::{db::HirDatabase, InferenceResult, Interner, Substitution, Ty, TyKind};

use self::pat_util::EnumerateAndAdjustIterator;

pub(crate) type PatId = Idx<Pat>;

#[derive(Clone, Debug)]
pub(crate) enum PatternError {
    Unimplemented,
    UnresolvedVariant,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FieldPat {
    pub(crate) field: LocalFieldId,
    pub(crate) pattern: Pat,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Pat {
    pub(crate) ty: Ty,
    pub(crate) kind: Box<PatKind>,
}

impl Pat {
    pub(crate) fn wildcard_from_ty(ty: &Ty) -> Self {
        Pat { ty: ty.clone(), kind: Box::new(PatKind::Wild) }
    }
}

/// Close relative to `rustc_mir_build::thir::pattern::PatKind`
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum PatKind {
    Wild,

    /// `x`, `ref x`, `x @ P`, etc.
    Binding {
        subpattern: Option<Pat>,
    },

    /// `Foo(...)` or `Foo{...}` or `Foo`, where `Foo` is a variant name from an ADT with
    /// multiple variants.
    Variant {
        substs: Substitution,
        enum_variant: EnumVariantId,
        subpatterns: Vec<FieldPat>,
    },

    /// `(...)`, `Foo(...)`, `Foo{...}`, or `Foo`, where `Foo` is a variant name from an ADT with
    /// a single variant.
    Leaf {
        subpatterns: Vec<FieldPat>,
    },

    /// `box P`, `&P`, `&mut P`, etc.
    Deref {
        subpattern: Pat,
    },

    // FIXME: for now, only bool literals are implemented
    LiteralBool {
        value: bool,
    },

    /// An or-pattern, e.g. `p | q`.
    /// Invariant: `pats.len() >= 2`.
    Or {
        pats: Vec<Pat>,
    },
}

pub(crate) struct PatCtxt<'a> {
    db: &'a dyn HirDatabase,
    infer: &'a InferenceResult,
    body: &'a Body,
    pub(crate) errors: Vec<PatternError>,
}

impl<'a> PatCtxt<'a> {
    pub(crate) fn new(db: &'a dyn HirDatabase, infer: &'a InferenceResult, body: &'a Body) -> Self {
        Self { db, infer, body, errors: Vec::new() }
    }

    pub(crate) fn lower_pattern(&mut self, pat: hir_def::expr::PatId) -> Pat {
        // FIXME: implement pattern adjustments (implicit pattern dereference; "RFC 2005-match-ergonomics")
        // More info https://github.com/rust-lang/rust/issues/42640#issuecomment-313535089
        let unadjusted_pat = self.lower_pattern_unadjusted(pat);
        unadjusted_pat
    }

    fn lower_pattern_unadjusted(&mut self, pat: hir_def::expr::PatId) -> Pat {
        let ty = &self.infer[pat];
        let variant = self.infer.variant_resolution_for_pat(pat);

        let kind = match self.body[pat] {
            hir_def::expr::Pat::Wild => PatKind::Wild,

            hir_def::expr::Pat::Lit(expr) => self.lower_lit(expr),

            hir_def::expr::Pat::Path(ref path) => {
                return self.lower_path(pat, path);
            }

            hir_def::expr::Pat::Tuple { ref args, ellipsis } => {
                let arity = match *ty.kind(&Interner) {
                    TyKind::Tuple(arity, _) => arity,
                    _ => panic!("unexpected type for tuple pattern: {:?}", ty),
                };
                let subpatterns = self.lower_tuple_subpats(args, arity, ellipsis);
                PatKind::Leaf { subpatterns }
            }

            hir_def::expr::Pat::Bind { subpat, .. } => {
                PatKind::Binding { subpattern: self.lower_opt_pattern(subpat) }
            }

            hir_def::expr::Pat::TupleStruct { ref args, ellipsis, .. } if variant.is_some() => {
                let expected_len = variant.unwrap().variant_data(self.db.upcast()).fields().len();
                let subpatterns = self.lower_tuple_subpats(args, expected_len, ellipsis);
                self.lower_variant_or_leaf(pat, ty, subpatterns)
            }

            hir_def::expr::Pat::Record { ref args, .. } if variant.is_some() => {
                let variant_data = variant.unwrap().variant_data(self.db.upcast());
                let subpatterns = args
                    .iter()
                    .map(|field| FieldPat {
                        // XXX(iDawer): field lookup is inefficient
                        field: variant_data.field(&field.name).unwrap(),
                        pattern: self.lower_pattern(field.pat),
                    })
                    .collect();
                self.lower_variant_or_leaf(pat, ty, subpatterns)
            }
            hir_def::expr::Pat::TupleStruct { .. } | hir_def::expr::Pat::Record { .. } => {
                self.errors.push(PatternError::UnresolvedVariant);
                PatKind::Wild
            }

            hir_def::expr::Pat::Or(ref pats) => PatKind::Or { pats: self.lower_patterns(pats) },

            _ => {
                self.errors.push(PatternError::Unimplemented);
                PatKind::Wild
            }
        };

        Pat { ty: ty.clone(), kind: Box::new(kind) }
    }

    fn lower_tuple_subpats(
        &mut self,
        pats: &[hir_def::expr::PatId],
        expected_len: usize,
        ellipsis: Option<usize>,
    ) -> Vec<FieldPat> {
        pats.iter()
            .enumerate_and_adjust(expected_len, ellipsis)
            .map(|(i, &subpattern)| FieldPat {
                field: LocalFieldId::from_raw((i as u32).into()),
                pattern: self.lower_pattern(subpattern),
            })
            .collect()
    }

    fn lower_patterns(&mut self, pats: &[hir_def::expr::PatId]) -> Vec<Pat> {
        pats.iter().map(|&p| self.lower_pattern(p)).collect()
    }

    fn lower_opt_pattern(&mut self, pat: Option<hir_def::expr::PatId>) -> Option<Pat> {
        pat.map(|p| self.lower_pattern(p))
    }

    fn lower_variant_or_leaf(
        &mut self,
        pat: hir_def::expr::PatId,
        ty: &Ty,
        subpatterns: Vec<FieldPat>,
    ) -> PatKind {
        let kind = match self.infer.variant_resolution_for_pat(pat) {
            Some(variant_id) => {
                if let VariantId::EnumVariantId(enum_variant) = variant_id {
                    let substs = match ty.kind(&Interner) {
                        TyKind::Adt(_, substs) | TyKind::FnDef(_, substs) => substs.clone(),
                        TyKind::Error => {
                            return PatKind::Wild;
                        }
                        _ => panic!("inappropriate type for def: {:?}", ty),
                    };
                    PatKind::Variant { substs, enum_variant, subpatterns }
                } else {
                    PatKind::Leaf { subpatterns }
                }
            }
            None => {
                self.errors.push(PatternError::UnresolvedVariant);
                PatKind::Wild
            }
        };
        kind
    }

    fn lower_path(&mut self, pat: hir_def::expr::PatId, _path: &hir_def::path::Path) -> Pat {
        let ty = &self.infer[pat];

        let pat_from_kind = |kind| Pat { ty: ty.clone(), kind: Box::new(kind) };

        match self.infer.variant_resolution_for_pat(pat) {
            Some(_) => pat_from_kind(self.lower_variant_or_leaf(pat, ty, Vec::new())),
            None => {
                self.errors.push(PatternError::UnresolvedVariant);
                pat_from_kind(PatKind::Wild)
            }
        }
    }

    fn lower_lit(&mut self, expr: hir_def::expr::ExprId) -> PatKind {
        use hir_def::expr::{Expr, Literal::Bool};

        match self.body[expr] {
            Expr::Literal(Bool(value)) => PatKind::LiteralBool { value },
            _ => {
                self.errors.push(PatternError::Unimplemented);
                PatKind::Wild
            }
        }
    }
}

pub(crate) trait PatternFoldable: Sized {
    fn fold_with<F: PatternFolder>(&self, folder: &mut F) -> Self {
        self.super_fold_with(folder)
    }

    fn super_fold_with<F: PatternFolder>(&self, folder: &mut F) -> Self;
}

pub(crate) trait PatternFolder: Sized {
    fn fold_pattern(&mut self, pattern: &Pat) -> Pat {
        pattern.super_fold_with(self)
    }

    fn fold_pattern_kind(&mut self, kind: &PatKind) -> PatKind {
        kind.super_fold_with(self)
    }
}

impl<T: PatternFoldable> PatternFoldable for Box<T> {
    fn super_fold_with<F: PatternFolder>(&self, folder: &mut F) -> Self {
        let content: T = (**self).fold_with(folder);
        Box::new(content)
    }
}

impl<T: PatternFoldable> PatternFoldable for Vec<T> {
    fn super_fold_with<F: PatternFolder>(&self, folder: &mut F) -> Self {
        self.iter().map(|t| t.fold_with(folder)).collect()
    }
}

impl<T: PatternFoldable> PatternFoldable for Option<T> {
    fn super_fold_with<F: PatternFolder>(&self, folder: &mut F) -> Self {
        self.as_ref().map(|t| t.fold_with(folder))
    }
}

macro_rules! clone_impls {
    ($($ty:ty),+) => {
        $(
            impl PatternFoldable for $ty {
                fn super_fold_with<F: PatternFolder>(&self, _: &mut F) -> Self {
                    Clone::clone(self)
                }
            }
        )+
    }
}

clone_impls! { LocalFieldId, Ty, Substitution, EnumVariantId }

impl PatternFoldable for FieldPat {
    fn super_fold_with<F: PatternFolder>(&self, folder: &mut F) -> Self {
        FieldPat { field: self.field.fold_with(folder), pattern: self.pattern.fold_with(folder) }
    }
}

impl PatternFoldable for Pat {
    fn fold_with<F: PatternFolder>(&self, folder: &mut F) -> Self {
        folder.fold_pattern(self)
    }

    fn super_fold_with<F: PatternFolder>(&self, folder: &mut F) -> Self {
        Pat { ty: self.ty.fold_with(folder), kind: self.kind.fold_with(folder) }
    }
}

impl PatternFoldable for PatKind {
    fn fold_with<F: PatternFolder>(&self, folder: &mut F) -> Self {
        folder.fold_pattern_kind(self)
    }

    fn super_fold_with<F: PatternFolder>(&self, folder: &mut F) -> Self {
        match self {
            PatKind::Wild => PatKind::Wild,
            PatKind::Binding { subpattern } => {
                PatKind::Binding { subpattern: subpattern.fold_with(folder) }
            }
            PatKind::Variant { substs, enum_variant, subpatterns } => PatKind::Variant {
                substs: substs.fold_with(folder),
                enum_variant: enum_variant.fold_with(folder),
                subpatterns: subpatterns.fold_with(folder),
            },
            PatKind::Leaf { subpatterns } => {
                PatKind::Leaf { subpatterns: subpatterns.fold_with(folder) }
            }
            PatKind::Deref { subpattern } => {
                PatKind::Deref { subpattern: subpattern.fold_with(folder) }
            }
            &PatKind::LiteralBool { value } => PatKind::LiteralBool { value },
            PatKind::Or { pats } => PatKind::Or { pats: pats.fold_with(folder) },
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::diagnostics::tests::check_diagnostics;

    #[test]
    fn unit() {
        check_diagnostics(
            r#"
fn main() {
    match () { () => {} }
    match () {  _ => {} }
    match () {          }
        //^^ Missing match arm
}
"#,
        );
    }

    #[test]
    fn tuple_of_units() {
        check_diagnostics(
            r#"
fn main() {
    match ((), ()) { ((), ()) => {} }
    match ((), ()) {  ((), _) => {} }
    match ((), ()) {   (_, _) => {} }
    match ((), ()) {        _ => {} }
    match ((), ()) {                }
        //^^^^^^^^ Missing match arm
}
"#,
        );
    }

    #[test]
    fn tuple_with_ellipsis() {
        check_diagnostics(
            r#"
struct A; struct B;
fn main(v: (A, (), B)) {
    match v { (A, ..)    => {} }
    match v { (.., B)    => {} }
    match v { (A, .., B) => {} }
    match v { (..)       => {} }
    match v {                  }
        //^ Missing match arm
}
"#,
        );
    }

    #[test]
    fn strukt() {
        check_diagnostics(
            r#"
struct A; struct B;
struct S { a: A, b: B}
fn main(v: S) {
    match v { S { a, b }       => {} }
    match v { S { a: _, b: _ } => {} }
    match v { S { .. }         => {} }
    match v { _                => {} }
    match v {                        }
        //^ Missing match arm
}
"#,
        );
    }

    #[test]
    fn c_enum() {
        check_diagnostics(
            r#"
enum E { A, B }
fn main(v: E) {
    match v { E::A | E::B => {} }
    match v { _           => {} }
    match v { E::A        => {} }
        //^ Missing match arm
    match v {                   }
        //^ Missing match arm
}
"#,
        );
    }

    #[test]
    fn enum_() {
        check_diagnostics(
            r#"
struct A; struct B;
enum E { Tuple(A, B), Struct{ a: A, b: B } }
fn main(v: E) {
    match v {
        E::Tuple(a, b)    => {}
        E::Struct{ a, b } => {}
    }
    match v {
        E::Tuple(_, _) => {}
        E::Struct{..}  => {}
    }
    match v {
        E::Tuple(..) => {}
        _ => {}
    }
    match v { E::Tuple(..) => {} }
        //^ Missing match arm
    match v { }
        //^ Missing match arm
}
"#,
        );
    }

    #[test]
    fn boolean() {
        check_diagnostics(
            r#"
fn main() {
    match true {
        true  => {}
        false => {}
    }
    match true {
        true | false => {}
    }
    match true {
        true => {}
        _ => {}
    }
    match true {}
        //^^^^ Missing match arm
    match true { true => {} }
        //^^^^ Missing match arm

}
"#,
        );
    }

    #[test]
    fn no_panic_at_unimplemented_subpattern_type() {
        check_diagnostics(
            r#"
struct S { a: char}
fn main(v: S) {
    match v { S{ a }      => {} }
    match v { S{ a: _x }   => {} }
    match v { S{ a: 'a' } => {} }
    match v { S{..}       => {} }
    match v { _           => {} }
    match v { }
        //^ Missing match arm
}
"#,
        );
    }

    #[test]
    fn binding() {
        check_diagnostics(
            r#"
fn main() {
    match true {
        _x @ true => {}
        false     => {}
    }
    match true { _x @ true => {} }
        //^^^^ Missing match arm
}
"#,
        );
    }
}
