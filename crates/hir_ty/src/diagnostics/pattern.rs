#![deny(elided_lifetimes_in_paths)]
#![allow(unused)] // todo remove

mod deconstruct_pat;
// TODO: find a better place for this?
mod pat_util;
pub mod usefulness;

use hir_def::{body::Body, EnumVariantId, LocalFieldId, VariantId};
use la_arena::Idx;

use crate::{db::HirDatabase, AdtId, InferenceResult, Interner, Substitution, Ty, TyKind};

use self::pat_util::EnumerateAndAdjustIterator;

pub type PatId = Idx<Pat>;

#[derive(Clone, Debug)]
pub(crate) enum PatternError {
    Unimplemented,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FieldPat {
    pub field: LocalFieldId,
    pub pattern: Pat,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Pat {
    pub ty: Ty,
    pub kind: Box<PatKind>,
}

impl Pat {
    pub(crate) fn wildcard_from_ty(ty: &Ty) -> Self {
        Pat { ty: ty.clone(), kind: Box::new(PatKind::Wild) }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum PatKind {
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

    // only bool for now
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
        // TODO: pattern adjustments (implicit dereference)
        // More info https://github.com/rust-lang/rust/issues/42640#issuecomment-313535089
        let unadjusted_pat = self.lower_pattern_unadjusted(pat);
        unadjusted_pat
    }

    fn lower_pattern_unadjusted(&mut self, pat: hir_def::expr::PatId) -> Pat {
        let ty = &self.infer[pat];

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

            hir_def::expr::Pat::TupleStruct { ref args, ellipsis, .. } => {
                let variant_data = match self.infer.variant_resolution_for_pat(pat) {
                    Some(variant_id) => variant_id.variant_data(self.db.upcast()),
                    None => panic!("tuple struct pattern not applied to an ADT {:?}", ty),
                };
                let subpatterns =
                    self.lower_tuple_subpats(args, variant_data.fields().len(), ellipsis);
                self.lower_variant_or_leaf(pat, ty, subpatterns)
            }

            hir_def::expr::Pat::Record { ref args, .. } => {
                let variant_data = match self.infer.variant_resolution_for_pat(pat) {
                    Some(variant_id) => variant_id.variant_data(self.db.upcast()),
                    None => panic!("record pattern not applied to an ADT {:?}", ty),
                };
                let subpatterns = args
                    .iter()
                    .map(|field| FieldPat {
                        // XXX(iDawer): field lookup is inefficient
                        field: variant_data.field(&field.name).unwrap_or_else(|| todo!()),
                        pattern: self.lower_pattern(field.pat),
                    })
                    .collect();
                self.lower_variant_or_leaf(pat, ty, subpatterns)
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
                self.errors.push(PatternError::Unimplemented);
                PatKind::Wild
            }
        };
        // TODO: do we need PatKind::AscribeUserType ?
        kind
    }

    fn lower_path(&mut self, pat: hir_def::expr::PatId, path: &hir_def::path::Path) -> Pat {
        let ty = &self.infer[pat];

        let pat_from_kind = |kind| Pat { ty: ty.clone(), kind: Box::new(kind) };

        match self.infer.variant_resolution_for_pat(pat) {
            Some(_) => pat_from_kind(self.lower_variant_or_leaf(pat, ty, Vec::new())),
            None => {
                self.errors.push(PatternError::Unimplemented);
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

#[cfg(test)]
mod tests {
    use crate::diagnostics::tests::check_diagnostics;

    use super::*;

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
        // TODO: test non-exhaustive match with ellipsis in the middle
        // of a pattern, check reported witness
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
    //FIXME: false negative. 
    // Binding patterns should be expanded in `usefulness::expand_pattern()`
    match true { _x @ true => {} }
}
"#,
        );
    }
}
