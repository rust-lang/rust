//! Validation of matches.
//!
//! This module provides lowering from [hir_def::expr::Pat] to [self::Pat] and match
//! checking algorithm.
//!
//! It is modeled on the rustc module `rustc_mir_build::thir::pattern`.

mod deconstruct_pat;
mod pat_util;
pub(crate) mod usefulness;

use hir_def::{body::Body, EnumVariantId, LocalFieldId, VariantId};
use la_arena::Idx;

use crate::{db::HirDatabase, InferenceResult, Interner, Substitution, Ty, TyKind};

use self::pat_util::EnumerateAndAdjustIterator;

pub(crate) use self::usefulness::MatchArm;

pub(crate) type PatId = Idx<Pat>;

#[derive(Clone, Debug)]
pub(crate) enum PatternError {
    Unimplemented,
    UnresolvedVariant,
    MissingField,
    ExtraFields,
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
    pub(crate) fn wildcard_from_ty(ty: Ty) -> Self {
        Pat { ty, kind: Box::new(PatKind::Wild) }
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
        // XXX(iDawer): Collecting pattern adjustments feels imprecise to me.
        // When lowering of & and box patterns are implemented this should be tested
        // in a manner of `match_ergonomics_issue_9095` test.
        // Pattern adjustment is part of RFC 2005-match-ergonomics.
        // More info https://github.com/rust-lang/rust/issues/42640#issuecomment-313535089
        let unadjusted_pat = self.lower_pattern_unadjusted(pat);
        self.infer.pat_adjustments.get(&pat).map(|it| &**it).unwrap_or_default().iter().rev().fold(
            unadjusted_pat,
            |subpattern, ref_ty| Pat {
                ty: ref_ty.clone(),
                kind: Box::new(PatKind::Deref { subpattern }),
            },
        )
    }

    fn lower_pattern_unadjusted(&mut self, pat: hir_def::expr::PatId) -> Pat {
        let mut ty = &self.infer[pat];
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
                if let TyKind::Ref(.., rty) = ty.kind(&Interner) {
                    ty = rty;
                }
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
                    .map(|field| {
                        // XXX(iDawer): field lookup is inefficient
                        variant_data.field(&field.name).map(|lfield_id| FieldPat {
                            field: lfield_id,
                            pattern: self.lower_pattern(field.pat),
                        })
                    })
                    .collect();
                match subpatterns {
                    Some(subpatterns) => self.lower_variant_or_leaf(pat, ty, subpatterns),
                    None => {
                        self.errors.push(PatternError::MissingField);
                        PatKind::Wild
                    }
                }
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
        if pats.len() > expected_len {
            self.errors.push(PatternError::ExtraFields);
            return Vec::new();
        }

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
pub(super) mod tests {
    mod report {
        use std::any::Any;

        use hir_def::{expr::PatId, DefWithBodyId};
        use hir_expand::{HirFileId, InFile};
        use syntax::SyntaxNodePtr;

        use crate::{
            db::HirDatabase,
            diagnostics_sink::{Diagnostic, DiagnosticCode, DiagnosticSink},
        };

        /// In tests, match check bails out loudly.
        /// This helps to catch incorrect tests that pass due to false negatives.
        pub(crate) fn report_bail_out(
            db: &dyn HirDatabase,
            def: DefWithBodyId,
            pat: PatId,
            sink: &mut DiagnosticSink,
        ) {
            let (_, source_map) = db.body_with_source_map(def);
            if let Ok(source_ptr) = source_map.pat_syntax(pat) {
                let pat_syntax_ptr = source_ptr.value.either(Into::into, Into::into);
                sink.push(BailedOut { file: source_ptr.file_id, pat_syntax_ptr });
            }
        }

        #[derive(Debug)]
        struct BailedOut {
            file: HirFileId,
            pat_syntax_ptr: SyntaxNodePtr,
        }

        impl Diagnostic for BailedOut {
            fn code(&self) -> DiagnosticCode {
                DiagnosticCode("internal:match-check-bailed-out")
            }
            fn message(&self) -> String {
                format!("Internal: match check bailed out")
            }
            fn display_source(&self) -> InFile<SyntaxNodePtr> {
                InFile { file_id: self.file, value: self.pat_syntax_ptr.clone() }
            }
            fn as_any(&self) -> &(dyn Any + Send + 'static) {
                self
            }
        }
    }

    use crate::diagnostics::tests::check_diagnostics;

    pub(crate) use self::report::report_bail_out;

    #[test]
    fn empty_tuple() {
        check_diagnostics(
            r#"
fn main() {
    match () { }
        //^^ Missing match arm
    match (()) { }
        //^^^^ Missing match arm

    match () { _ => (), }
    match () { () => (), }
    match (()) { (()) => (), }
}
"#,
        );
    }

    #[test]
    fn tuple_of_two_empty_tuple() {
        check_diagnostics(
            r#"
fn main() {
    match ((), ()) { }
        //^^^^^^^^ Missing match arm

    match ((), ()) { ((), ()) => (), }
}
"#,
        );
    }

    #[test]
    fn boolean() {
        check_diagnostics(
            r#"
fn test_main() {
    match false { }
        //^^^^^ Missing match arm
    match false { true => (), }
        //^^^^^ Missing match arm
    match (false, true) {}
        //^^^^^^^^^^^^^ Missing match arm
    match (false, true) { (true, true) => (), }
        //^^^^^^^^^^^^^ Missing match arm
    match (false, true) {
        //^^^^^^^^^^^^^ Missing match arm
        (false, true) => (),
        (false, false) => (),
        (true, false) => (),
    }
    match (false, true) { (true, _x) => (), }
        //^^^^^^^^^^^^^ Missing match arm

    match false { true => (), false => (), }
    match (false, true) {
        (false, _) => (),
        (true, false) => (),
        (_, true) => (),
    }
    match (false, true) {
        (true, true) => (),
        (true, false) => (),
        (false, true) => (),
        (false, false) => (),
    }
    match (false, true) {
        (true, _x) => (),
        (false, true) => (),
        (false, false) => (),
    }
    match (false, true, false) {
        (false, ..) => (),
        (true, ..) => (),
    }
    match (false, true, false) {
        (.., false) => (),
        (.., true) => (),
    }
    match (false, true, false) { (..) => (), }
}
"#,
        );
    }

    #[test]
    fn tuple_of_tuple_and_bools() {
        check_diagnostics(
            r#"
fn main() {
    match (false, ((), false)) {}
        //^^^^^^^^^^^^^^^^^^^^ Missing match arm
    match (false, ((), false)) { (true, ((), true)) => (), }
        //^^^^^^^^^^^^^^^^^^^^ Missing match arm
    match (false, ((), false)) { (true, _) => (), }
        //^^^^^^^^^^^^^^^^^^^^ Missing match arm

    match (false, ((), false)) {
        (true, ((), true)) => (),
        (true, ((), false)) => (),
        (false, ((), true)) => (),
        (false, ((), false)) => (),
    }
    match (false, ((), false)) {
        (true, ((), true)) => (),
        (true, ((), false)) => (),
        (false, _) => (),
    }
}
"#,
        );
    }

    #[test]
    fn enums() {
        check_diagnostics(
            r#"
enum Either { A, B, }

fn main() {
    match Either::A { }
        //^^^^^^^^^ Missing match arm
    match Either::B { Either::A => (), }
        //^^^^^^^^^ Missing match arm

    match &Either::B {
        //^^^^^^^^^^ Missing match arm
        Either::A => (),
    }

    match Either::B {
        Either::A => (), Either::B => (),
    }
    match &Either::B {
        Either::A => (), Either::B => (),
    }
}
"#,
        );
    }

    #[test]
    fn enum_containing_bool() {
        check_diagnostics(
            r#"
enum Either { A(bool), B }

fn main() {
    match Either::B { }
        //^^^^^^^^^ Missing match arm
    match Either::B {
        //^^^^^^^^^ Missing match arm
        Either::A(true) => (), Either::B => ()
    }

    match Either::B {
        Either::A(true) => (),
        Either::A(false) => (),
        Either::B => (),
    }
    match Either::B {
        Either::B => (),
        _ => (),
    }
    match Either::B {
        Either::A(_) => (),
        Either::B => (),
    }

}
        "#,
        );
    }

    #[test]
    fn enum_different_sizes() {
        check_diagnostics(
            r#"
enum Either { A(bool), B(bool, bool) }

fn main() {
    match Either::A(false) {
        //^^^^^^^^^^^^^^^^ Missing match arm
        Either::A(_) => (),
        Either::B(false, _) => (),
    }

    match Either::A(false) {
        Either::A(_) => (),
        Either::B(true, _) => (),
        Either::B(false, _) => (),
    }
    match Either::A(false) {
        Either::A(true) | Either::A(false) => (),
        Either::B(true, _) => (),
        Either::B(false, _) => (),
    }
}
"#,
        );
    }

    #[test]
    fn tuple_of_enum_no_diagnostic() {
        check_diagnostics(
            r#"
enum Either { A(bool), B(bool, bool) }
enum Either2 { C, D }

fn main() {
    match (Either::A(false), Either2::C) {
        (Either::A(true), _) | (Either::A(false), _) => (),
        (Either::B(true, _), Either2::C) => (),
        (Either::B(false, _), Either2::C) => (),
        (Either::B(_, _), Either2::D) => (),
    }
}
"#,
        );
    }

    #[test]
    fn or_pattern_no_diagnostic() {
        check_diagnostics(
            r#"
enum Either {A, B}

fn main() {
    match (Either::A, Either::B) {
        (Either::A | Either::B, _) => (),
    }
}"#,
        )
    }

    #[test]
    fn mismatched_types() {
        // Match statements with arms that don't match the
        // expression pattern do not fire this diagnostic.
        check_diagnostics(
            r#"
enum Either { A, B }
enum Either2 { C, D }

fn main() {
    match Either::A {
        Either2::C => (),
    //  ^^^^^^^^^^ Internal: match check bailed out
        Either2::D => (),
    }
    match (true, false) {
        (true, false, true) => (),
    //  ^^^^^^^^^^^^^^^^^^^ Internal: match check bailed out
        (true) => (),
    }
    match (true, false) { (true,) => {} }
    //                    ^^^^^^^ Internal: match check bailed out
    match (0) { () => () }
            //  ^^ Internal: match check bailed out
    match Unresolved::Bar { Unresolved::Baz => () }
}
        "#,
        );
    }

    #[test]
    fn mismatched_types_in_or_patterns() {
        check_diagnostics(
            r#"
fn main() {
    match false { true | () => {} }
    //            ^^^^^^^^^ Internal: match check bailed out
    match (false,) { (true | (),) => {} }
    //               ^^^^^^^^^^^^ Internal: match check bailed out
}
"#,
        );
    }

    #[test]
    fn malformed_match_arm_tuple_enum_missing_pattern() {
        // We are testing to be sure we don't panic here when the match
        // arm `Either::B` is missing its pattern.
        check_diagnostics(
            r#"
enum Either { A, B(u32) }

fn main() {
    match Either::A {
        Either::A => (),
        Either::B() => (),
    }
}
"#,
        );
    }

    #[test]
    fn malformed_match_arm_extra_fields() {
        check_diagnostics(
            r#"
enum A { B(isize, isize), C }
fn main() {
    match A::B(1, 2) {
        A::B(_, _, _) => (),
    //  ^^^^^^^^^^^^^ Internal: match check bailed out
    }
    match A::B(1, 2) {
        A::C(_) => (),
    //  ^^^^^^^ Internal: match check bailed out
    }
}
"#,
        );
    }

    #[test]
    fn expr_diverges() {
        check_diagnostics(
            r#"
enum Either { A, B }

fn main() {
    match loop {} {
        Either::A => (),
    //  ^^^^^^^^^ Internal: match check bailed out
        Either::B => (),
    }
    match loop {} {
        Either::A => (),
    //  ^^^^^^^^^ Internal: match check bailed out
    }
    match loop { break Foo::A } {
        //^^^^^^^^^^^^^^^^^^^^^ Missing match arm
        Either::A => (),
    }
    match loop { break Foo::A } {
        Either::A => (),
        Either::B => (),
    }
}
"#,
        );
    }

    #[test]
    fn expr_partially_diverges() {
        check_diagnostics(
            r#"
enum Either<T> { A(T), B }

fn foo() -> Either<!> { Either::B }
fn main() -> u32 {
    match foo() {
        Either::A(val) => val,
        Either::B => 0,
    }
}
"#,
        );
    }

    #[test]
    fn enum_record() {
        check_diagnostics(
            r#"
enum Either { A { foo: bool }, B }

fn main() {
    let a = Either::A { foo: true };
    match a { }
        //^ Missing match arm
    match a { Either::A { foo: true } => () }
        //^ Missing match arm
    match a {
        Either::A { } => (),
      //^^^^^^^^^ Missing structure fields:
      //        | - foo
        Either::B => (),
    }
    match a {
        //^ Missing match arm
        Either::A { } => (),
    } //^^^^^^^^^ Missing structure fields:
      //        | - foo

    match a {
        Either::A { foo: true } => (),
        Either::A { foo: false } => (),
        Either::B => (),
    }
    match a {
        Either::A { foo: _ } => (),
        Either::B => (),
    }
}
"#,
        );
    }

    #[test]
    fn enum_record_fields_out_of_order() {
        check_diagnostics(
            r#"
enum Either {
    A { foo: bool, bar: () },
    B,
}

fn main() {
    let a = Either::A { foo: true, bar: () };
    match a {
        //^ Missing match arm
        Either::A { bar: (), foo: false } => (),
        Either::A { foo: true, bar: () } => (),
    }

    match a {
        Either::A { bar: (), foo: false } => (),
        Either::A { foo: true, bar: () } => (),
        Either::B => (),
    }
}
"#,
        );
    }

    #[test]
    fn enum_record_ellipsis() {
        check_diagnostics(
            r#"
enum Either {
    A { foo: bool, bar: bool },
    B,
}

fn main() {
    let a = Either::B;
    match a {
        //^ Missing match arm
        Either::A { foo: true, .. } => (),
        Either::B => (),
    }
    match a {
        //^ Missing match arm
        Either::A { .. } => (),
    }

    match a {
        Either::A { foo: true, .. } => (),
        Either::A { foo: false, .. } => (),
        Either::B => (),
    }

    match a {
        Either::A { .. } => (),
        Either::B => (),
    }
}
"#,
        );
    }

    #[test]
    fn enum_tuple_partial_ellipsis() {
        check_diagnostics(
            r#"
enum Either {
    A(bool, bool, bool, bool),
    B,
}

fn main() {
    match Either::B {
        //^^^^^^^^^ Missing match arm
        Either::A(true, .., true) => (),
        Either::A(true, .., false) => (),
        Either::A(false, .., false) => (),
        Either::B => (),
    }
    match Either::B {
        //^^^^^^^^^ Missing match arm
        Either::A(true, .., true) => (),
        Either::A(true, .., false) => (),
        Either::A(.., true) => (),
        Either::B => (),
    }

    match Either::B {
        Either::A(true, .., true) => (),
        Either::A(true, .., false) => (),
        Either::A(false, .., true) => (),
        Either::A(false, .., false) => (),
        Either::B => (),
    }
    match Either::B {
        Either::A(true, .., true) => (),
        Either::A(true, .., false) => (),
        Either::A(.., true) => (),
        Either::A(.., false) => (),
        Either::B => (),
    }
}
"#,
        );
    }

    #[test]
    fn never() {
        check_diagnostics(
            r#"
enum Never {}

fn enum_(never: Never) {
    match never {}
}
fn enum_ref(never: &Never) {
    match never {}
        //^^^^^ Missing match arm
}
fn bang(never: !) {
    match never {}
}
"#,
        );
    }

    #[test]
    fn unknown_type() {
        check_diagnostics(
            r#"
enum Option<T> { Some(T), None }

fn main() {
    // `Never` is deliberately not defined so that it's an uninferred type.
    match Option::<Never>::None {
        None => (),
        Some(never) => match never {},
    //  ^^^^^^^^^^^ Internal: match check bailed out
    }
    match Option::<Never>::None {
        //^^^^^^^^^^^^^^^^^^^^^ Missing match arm
        Option::Some(_never) => {},
    }
}
"#,
        );
    }

    #[test]
    fn tuple_of_bools_with_ellipsis_at_end_missing_arm() {
        check_diagnostics(
            r#"
fn main() {
    match (false, true, false) {
        //^^^^^^^^^^^^^^^^^^^^ Missing match arm
        (false, ..) => (),
    }
}"#,
        );
    }

    #[test]
    fn tuple_of_bools_with_ellipsis_at_beginning_missing_arm() {
        check_diagnostics(
            r#"
fn main() {
    match (false, true, false) {
        //^^^^^^^^^^^^^^^^^^^^ Missing match arm
        (.., false) => (),
    }
}"#,
        );
    }

    #[test]
    fn tuple_of_bools_with_ellipsis_in_middle_missing_arm() {
        check_diagnostics(
            r#"
fn main() {
    match (false, true, false) {
        //^^^^^^^^^^^^^^^^^^^^ Missing match arm
        (true, .., false) => (),
    }
}"#,
        );
    }

    #[test]
    fn record_struct() {
        check_diagnostics(
            r#"struct Foo { a: bool }
fn main(f: Foo) {
    match f {}
        //^ Missing match arm
    match f { Foo { a: true } => () }
        //^ Missing match arm
    match &f { Foo { a: true } => () }
        //^^ Missing match arm
    match f { Foo { a: _ } => () }
    match f {
        Foo { a: true } => (),
        Foo { a: false } => (),
    }
    match &f {
        Foo { a: true } => (),
        Foo { a: false } => (),
    }
}
"#,
        );
    }

    #[test]
    fn tuple_struct() {
        check_diagnostics(
            r#"struct Foo(bool);
fn main(f: Foo) {
    match f {}
        //^ Missing match arm
    match f { Foo(true) => () }
        //^ Missing match arm
    match f {
        Foo(true) => (),
        Foo(false) => (),
    }
}
"#,
        );
    }

    #[test]
    fn unit_struct() {
        check_diagnostics(
            r#"struct Foo;
fn main(f: Foo) {
    match f {}
        //^ Missing match arm
    match f { Foo => () }
}
"#,
        );
    }

    #[test]
    fn record_struct_ellipsis() {
        check_diagnostics(
            r#"struct Foo { foo: bool, bar: bool }
fn main(f: Foo) {
    match f { Foo { foo: true, .. } => () }
        //^ Missing match arm
    match f {
        //^ Missing match arm
        Foo { foo: true, .. } => (),
        Foo { bar: false, .. } => ()
    }
    match f { Foo { .. } => () }
    match f {
        Foo { foo: true, .. } => (),
        Foo { foo: false, .. } => ()
    }
}
"#,
        );
    }

    #[test]
    fn internal_or() {
        check_diagnostics(
            r#"
fn main() {
    enum Either { A(bool), B }
    match Either::B {
        //^^^^^^^^^ Missing match arm
        Either::A(true | false) => (),
    }
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
    match v { S{ a: _x }  => {} }
    match v { S{ a: 'a' } => {} }
            //^^^^^^^^^^^ Internal: match check bailed out
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

    #[test]
    fn binding_ref_has_correct_type() {
        // Asserts `PatKind::Binding(ref _x): bool`, not &bool.
        // If that's not true match checking will panic with "incompatible constructors"
        // FIXME: make facilities to test this directly like `tests::check_infer(..)`
        check_diagnostics(
            r#"
enum Foo { A }
fn main() {
    // FIXME: this should not bail out but current behavior is such as the old algorithm.
    // ExprValidator::validate_match(..) checks types of top level patterns incorrecly.
    match Foo::A {
        ref _x => {}
    //  ^^^^^^ Internal: match check bailed out
        Foo::A => {}
    }
    match (true,) {
        (ref _x,) => {}
        (true,) => {}
    }
}
"#,
        );
    }

    #[test]
    fn enum_non_exhaustive() {
        check_diagnostics(
            r#"
//- /lib.rs crate:lib
#[non_exhaustive]
pub enum E { A, B }
fn _local() {
    match E::A { _ => {} }
    match E::A {
        E::A => {}
        E::B => {}
    }
    match E::A {
        E::A | E::B => {}
    }
}

//- /main.rs crate:main deps:lib
use lib::E;
fn main() {
    match E::A { _ => {} }
    match E::A {
        //^^^^ Missing match arm
        E::A => {}
        E::B => {}
    }
    match E::A {
        //^^^^ Missing match arm
        E::A | E::B => {}
    }
}
"#,
        );
    }

    #[test]
    fn match_guard() {
        check_diagnostics(
            r#"
fn main() {
    match true {
        true if false => {}
        true          => {}
        false         => {}
    }
    match true {
        //^^^^ Missing match arm
        true if false => {}
        false         => {}
}
"#,
        );
    }

    #[test]
    fn pattern_type_is_of_substitution() {
        cov_mark::check!(match_check_wildcard_expanded_to_substitutions);
        check_diagnostics(
            r#"
struct Foo<T>(T);
struct Bar;
fn main() {
    match Foo(Bar) {
        _ | Foo(Bar) => {}
    }
}
"#,
        );
    }

    #[test]
    fn record_struct_no_such_field() {
        check_diagnostics(
            r#"
struct Foo { }
fn main(f: Foo) {
    match f { Foo { bar } => () }
    //        ^^^^^^^^^^^ Internal: match check bailed out
}
"#,
        );
    }

    #[test]
    fn match_ergonomics_issue_9095() {
        check_diagnostics(
            r#"
enum Foo<T> { A(T) }
fn main() {
    match &Foo::A(true) {
        _ => {}
        Foo::A(_) => {}
    }
}
"#,
        );
    }

    mod false_negatives {
        //! The implementation of match checking here is a work in progress. As we roll this out, we
        //! prefer false negatives to false positives (ideally there would be no false positives). This
        //! test module should document known false negatives. Eventually we will have a complete
        //! implementation of match checking and this module will be empty.
        //!
        //! The reasons for documenting known false negatives:
        //!
        //!   1. It acts as a backlog of work that can be done to improve the behavior of the system.
        //!   2. It ensures the code doesn't panic when handling these cases.
        use super::*;

        #[test]
        fn integers() {
            // We don't currently check integer exhaustiveness.
            check_diagnostics(
                r#"
fn main() {
    match 5 {
        10 => (),
    //  ^^ Internal: match check bailed out
        11..20 => (),
    }
}
"#,
            );
        }

        #[test]
        fn reference_patterns_at_top_level() {
            check_diagnostics(
                r#"
fn main() {
    match &false {
        &true => {}
    //  ^^^^^ Internal: match check bailed out
    }
}
            "#,
            );
        }

        #[test]
        fn reference_patterns_in_fields() {
            check_diagnostics(
                r#"
fn main() {
    match (&false,) {
        (true,) => {}
    //  ^^^^^^^ Internal: match check bailed out
    }
    match (&false,) {
        (&true,) => {}
    //  ^^^^^^^^ Internal: match check bailed out
    }
}
            "#,
            );
        }
    }
}
