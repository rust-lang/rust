//! This module implements match statement exhaustiveness checking and usefulness checking
//! for match arms.
//!
//! It is modeled on the rustc module `librustc_mir_build::hair::pattern::_match`, which
//! contains very detailed documentation about the match checking algorithm.
use std::sync::Arc;

use smallvec::{smallvec, SmallVec};

use crate::{
    db::HirDatabase,
    expr::{Body, Expr, Literal, Pat, PatId},
    InferenceResult,
};
use hir_def::{adt::VariantData, EnumVariantId, VariantId};

#[derive(Debug, Clone, Copy)]
enum PatIdOrWild {
    PatId(PatId),
    Wild,
}

impl PatIdOrWild {
    fn as_pat(self, cx: &MatchCheckCtx) -> Pat {
        match self {
            PatIdOrWild::PatId(id) => cx.body.pats[id].clone(),
            PatIdOrWild::Wild => Pat::Wild,
        }
    }

    fn as_id(self) -> Option<PatId> {
        match self {
            PatIdOrWild::PatId(id) => Some(id),
            PatIdOrWild::Wild => None,
        }
    }
}

impl From<PatId> for PatIdOrWild {
    fn from(pat_id: PatId) -> Self {
        Self::PatId(pat_id)
    }
}

type PatStackInner = SmallVec<[PatIdOrWild; 2]>;
#[derive(Debug)]
pub(crate) struct PatStack(PatStackInner);

impl PatStack {
    pub(crate) fn from_pattern(pat_id: PatId) -> PatStack {
        Self(smallvec!(pat_id.into()))
    }

    pub(crate) fn from_wild() -> PatStack {
        Self(smallvec!(PatIdOrWild::Wild))
    }

    fn from_slice(slice: &[PatIdOrWild]) -> PatStack {
        Self(SmallVec::from_slice(slice))
    }

    fn from_vec(v: PatStackInner) -> PatStack {
        Self(v)
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn head(&self) -> PatIdOrWild {
        self.0[0]
    }

    fn get_head(&self) -> Option<PatIdOrWild> {
        self.0.first().copied()
    }

    fn to_tail(&self) -> PatStack {
        Self::from_slice(&self.0[1..])
    }

    fn replace_head_with(&self, pat_ids: &[PatId]) -> PatStack {
        let mut patterns: PatStackInner = smallvec![];
        for pat in pat_ids {
            patterns.push((*pat).into());
        }
        for pat in &self.0[1..] {
            patterns.push(*pat);
        }
        PatStack::from_vec(patterns)
    }

    // Computes `D(self)`.
    fn specialize_wildcard(&self, cx: &MatchCheckCtx) -> Option<PatStack> {
        if matches!(self.head().as_pat(cx), Pat::Wild) {
            Some(self.to_tail())
        } else {
            None
        }
    }

    // Computes `S(constructor, self)`.
    fn specialize_constructor(
        &self,
        cx: &MatchCheckCtx,
        constructor: &Constructor,
    ) -> Option<PatStack> {
        match (self.head().as_pat(cx), constructor) {
            (Pat::Tuple(ref pat_ids), Constructor::Tuple { arity }) => {
                if pat_ids.len() != *arity {
                    return None;
                }

                Some(self.replace_head_with(pat_ids))
            }
            (Pat::Lit(_), Constructor::Bool(_)) => {
                // for now we only support bool literals
                Some(self.to_tail())
            }
            (Pat::Wild, constructor) => Some(self.expand_wildcard(cx, constructor)),
            (Pat::Path(_), Constructor::Enum(constructor)) => {
                let pat_id = self.head().as_id().expect("we know this isn't a wild");
                if !enum_variant_matches(cx, pat_id, *constructor) {
                    return None;
                }
                // enums with no associated data become `Pat::Path`
                Some(self.to_tail())
            }
            (Pat::TupleStruct { args: ref pat_ids, .. }, Constructor::Enum(constructor)) => {
                let pat_id = self.head().as_id().expect("we know this isn't a wild");
                if !enum_variant_matches(cx, pat_id, *constructor) {
                    return None;
                }

                Some(self.replace_head_with(pat_ids))
            }
            (Pat::Or(_), _) => unreachable!("we desugar or patterns so this should never happen"),
            (a, b) => unimplemented!("{:?}, {:?}", a, b),
        }
    }

    fn expand_wildcard(&self, cx: &MatchCheckCtx, constructor: &Constructor) -> PatStack {
        assert_eq!(
            Pat::Wild,
            self.head().as_pat(cx),
            "expand_wildcard must only be called on PatStack with wild at head",
        );

        let mut patterns: PatStackInner = smallvec![];
        let arity = match constructor {
            Constructor::Bool(_) => 0,
            Constructor::Tuple { arity } => *arity,
            Constructor::Enum(e) => {
                match cx.db.enum_data(e.parent).variants[e.local_id].variant_data.as_ref() {
                    VariantData::Tuple(struct_field_data) => struct_field_data.len(),
                    VariantData::Unit => 0,
                    x => unimplemented!("{:?}", x),
                }
            }
        };

        for _ in 0..arity {
            patterns.push(PatIdOrWild::Wild);
        }

        for pat in &self.0[1..] {
            patterns.push(*pat);
        }

        PatStack::from_vec(patterns)
    }
}

#[derive(Debug)]
pub(crate) struct Matrix(Vec<PatStack>);

impl Matrix {
    pub(crate) fn empty() -> Self {
        Self(vec![])
    }

    pub(crate) fn push(&mut self, cx: &MatchCheckCtx, row: PatStack) {
        // if the pattern is an or pattern it should be expanded
        if let Some(Pat::Or(pat_ids)) = row.get_head().map(|pat_id| pat_id.as_pat(cx)) {
            for pat_id in pat_ids {
                self.0.push(PatStack::from_pattern(pat_id));
            }
        } else {
            self.0.push(row);
        }
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn heads(&self) -> Vec<PatIdOrWild> {
        self.0.iter().map(|p| p.head()).collect()
    }

    // Computes `D(self)`.
    fn specialize_wildcard(&self, cx: &MatchCheckCtx) -> Self {
        Self::collect(cx, self.0.iter().filter_map(|r| r.specialize_wildcard(cx)))
    }

    // Computes `S(constructor, self)`.
    fn specialize_constructor(&self, cx: &MatchCheckCtx, constructor: &Constructor) -> Self {
        Self::collect(cx, self.0.iter().filter_map(|r| r.specialize_constructor(cx, constructor)))
    }

    fn collect<T: IntoIterator<Item = PatStack>>(cx: &MatchCheckCtx, iter: T) -> Self {
        let mut matrix = Matrix::empty();

        for pat in iter {
            // using push ensures we expand or-patterns
            matrix.push(cx, pat);
        }

        matrix
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Usefulness {
    Useful,
    NotUseful,
}

pub struct MatchCheckCtx<'a> {
    pub body: Arc<Body>,
    pub match_expr: &'a Expr,
    pub infer: Arc<InferenceResult>,
    pub db: &'a dyn HirDatabase,
}

// see src/librustc_mir_build/hair/pattern/_match.rs
// It seems the rustc version of this method is able to assume that all the match arm
// patterns are valid (they are valid given a particular match expression), but I
// don't think we can make that assumption here. How should that be handled?
//
// Perhaps check that validity before passing the patterns into this method?
pub(crate) fn is_useful(cx: &MatchCheckCtx, matrix: &Matrix, v: &PatStack) -> Usefulness {
    dbg!(matrix);
    dbg!(v);
    if v.is_empty() {
        if matrix.is_empty() {
            return Usefulness::Useful;
        } else {
            return Usefulness::NotUseful;
        }
    }

    if let Pat::Or(pat_ids) = v.head().as_pat(cx) {
        let any_useful = pat_ids.iter().any(|&pat_id| {
            let v = PatStack::from_pattern(pat_id);

            is_useful(cx, matrix, &v) == Usefulness::Useful
        });

        return if any_useful { Usefulness::Useful } else { Usefulness::NotUseful };
    }

    if let Some(constructor) = pat_constructor(cx, v.head()) {
        let matrix = matrix.specialize_constructor(&cx, &constructor);
        let v = v.specialize_constructor(&cx, &constructor).expect("todo handle this case");

        is_useful(&cx, &matrix, &v)
    } else {
        dbg!("expanding wildcard");
        // expanding wildcard
        let used_constructors: Vec<Constructor> =
            matrix.heads().iter().filter_map(|&p| pat_constructor(cx, p)).collect();

        // We assume here that the first constructor is the "correct" type. Since we
        // only care about the "type" of the constructor (i.e. if it is a bool we
        // don't care about the value), this assumption should be valid as long as
        // the match statement is well formed. But potentially a better way to handle
        // this is to use the match expressions type.
        match &used_constructors.first() {
            Some(constructor) if all_constructors_covered(&cx, constructor, &used_constructors) => {
                dbg!("all constructors are covered");
                // If all constructors are covered, then we need to consider whether
                // any values are covered by this wildcard.
                //
                // For example, with matrix '[[Some(true)], [None]]', all
                // constructors are covered (`Some`/`None`), so we need
                // to perform specialization to see that our wildcard will cover
                // the `Some(false)` case.
                let constructor =
                    matrix.heads().iter().filter_map(|&pat| pat_constructor(cx, pat)).next();

                if let Some(constructor) = constructor {
                    dbg!("found constructor {:?}, specializing..", &constructor);
                    if let Constructor::Enum(e) = constructor {
                        // For enums we handle each variant as a distinct constructor, so
                        // here we create a constructor for each variant and then check
                        // usefulness after specializing for that constructor.
                        let any_useful = cx
                            .db
                            .enum_data(e.parent)
                            .variants
                            .iter()
                            .map(|(local_id, _)| {
                                Constructor::Enum(EnumVariantId { parent: e.parent, local_id })
                            })
                            .any(|constructor| {
                                let matrix = matrix.specialize_constructor(&cx, &constructor);
                                let v = v.expand_wildcard(&cx, &constructor);

                                is_useful(&cx, &matrix, &v) == Usefulness::Useful
                            });

                        if any_useful {
                            Usefulness::Useful
                        } else {
                            Usefulness::NotUseful
                        }
                    } else {
                        let matrix = matrix.specialize_constructor(&cx, &constructor);
                        let v = v.expand_wildcard(&cx, &constructor);

                        is_useful(&cx, &matrix, &v)
                    }
                } else {
                    Usefulness::NotUseful
                }
            }
            _ => {
                // Either not all constructors are covered, or the only other arms
                // are wildcards. Either way, this pattern is useful if it is useful
                // when compared to those arms with wildcards.
                let matrix = matrix.specialize_wildcard(&cx);
                let v = v.to_tail();

                is_useful(&cx, &matrix, &v)
            }
        }
    }
}

#[derive(Debug)]
enum Constructor {
    Bool(bool),
    Tuple { arity: usize },
    Enum(EnumVariantId),
}

fn pat_constructor(cx: &MatchCheckCtx, pat: PatIdOrWild) -> Option<Constructor> {
    match pat.as_pat(cx) {
        Pat::Wild => None,
        Pat::Tuple(pats) => Some(Constructor::Tuple { arity: pats.len() }),
        Pat::Lit(lit_expr) => {
            // for now we only support bool literals
            match cx.body.exprs[lit_expr] {
                Expr::Literal(Literal::Bool(val)) => Some(Constructor::Bool(val)),
                _ => unimplemented!(),
            }
        }
        Pat::TupleStruct { .. } | Pat::Path(_) => {
            let pat_id = pat.as_id().expect("we already know this pattern is not a wild");
            let variant_id =
                cx.infer.variant_resolution_for_pat(pat_id).unwrap_or_else(|| unimplemented!());
            match variant_id {
                VariantId::EnumVariantId(enum_variant_id) => {
                    Some(Constructor::Enum(enum_variant_id))
                }
                _ => unimplemented!(),
            }
        }
        x => unimplemented!("{:?} not yet implemented", x),
    }
}

fn all_constructors_covered(
    cx: &MatchCheckCtx,
    constructor: &Constructor,
    used_constructors: &[Constructor],
) -> bool {
    match constructor {
        Constructor::Tuple { arity } => {
            used_constructors.iter().any(|constructor| match constructor {
                Constructor::Tuple { arity: used_arity } => arity == used_arity,
                _ => false,
            })
        }
        Constructor::Bool(_) => {
            if used_constructors.is_empty() {
                return false;
            }

            let covers_true =
                used_constructors.iter().any(|c| matches!(c, Constructor::Bool(true)));
            let covers_false =
                used_constructors.iter().any(|c| matches!(c, Constructor::Bool(false)));

            covers_true && covers_false
        }
        Constructor::Enum(e) => cx.db.enum_data(e.parent).variants.iter().all(|(id, _)| {
            for constructor in used_constructors {
                if let Constructor::Enum(e) = constructor {
                    if id == e.local_id {
                        return true;
                    }
                }
            }

            false
        }),
    }
}

fn enum_variant_matches(cx: &MatchCheckCtx, pat_id: PatId, enum_variant_id: EnumVariantId) -> bool {
    if let Some(VariantId::EnumVariantId(pat_variant_id)) =
        cx.infer.variant_resolution_for_pat(pat_id)
    {
        if pat_variant_id.local_id == enum_variant_id.local_id {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    pub(super) use insta::assert_snapshot;
    pub(super) use ra_db::fixture::WithFixture;

    pub(super) use crate::test_db::TestDB;

    pub(super) fn check_diagnostic_message(content: &str) -> String {
        TestDB::with_single_file(content).0.diagnostics().0
    }

    pub(super) fn check_diagnostic_with_no_fix(content: &str) {
        let diagnostic_count = TestDB::with_single_file(content).0.diagnostics().1;

        assert_eq!(1, diagnostic_count, "no diagnotic reported");
    }

    pub(super) fn check_no_diagnostic(content: &str) {
        let diagnostic_count = TestDB::with_single_file(content).0.diagnostics().1;

        assert_eq!(0, diagnostic_count, "expected no diagnostic, found one");
    }

    #[test]
    fn empty_tuple_no_arms_diagnostic_message() {
        let content = r"
            fn test_fn() {
                match () {
                }
            }
        ";

        assert_snapshot!(
            check_diagnostic_message(content),
            @"\"{\\n                }\": Missing match arm\n"
        );
    }

    #[test]
    fn empty_tuple_no_arms() {
        let content = r"
            fn test_fn() {
                match () {
                }
            }
        ";

        check_diagnostic_with_no_fix(content);
    }

    #[test]
    fn empty_tuple_no_diagnostic() {
        let content = r"
            fn test_fn() {
                match () {
                    () => {}
                }
            }
        ";

        check_no_diagnostic(content);
    }

    #[test]
    fn tuple_of_empty_tuple_no_arms() {
        let content = r"
            fn test_fn() {
                match (()) {
                }
            }
        ";

        check_diagnostic_with_no_fix(content);
    }

    #[test]
    fn tuple_of_empty_tuple_no_diagnostic() {
        let content = r"
            fn test_fn() {
                match (()) {
                    (()) => {}
                }
            }
        ";

        check_no_diagnostic(content);
    }

    #[test]
    fn tuple_of_two_empty_tuple_no_arms() {
        let content = r"
            fn test_fn() {
                match ((), ()) {
                }
            }
        ";

        check_diagnostic_with_no_fix(content);
    }

    #[test]
    fn tuple_of_two_empty_tuple_no_diagnostic() {
        let content = r"
            fn test_fn() {
                match ((), ()) {
                    ((), ()) => {}
                }
            }
        ";

        check_no_diagnostic(content);
    }

    #[test]
    fn bool_no_arms() {
        let content = r"
            fn test_fn() {
                match false {
                }
            }
        ";

        check_diagnostic_with_no_fix(content);
    }

    #[test]
    fn bool_missing_arm() {
        let content = r"
            fn test_fn() {
                match false {
                    true => {}
                }
            }
        ";

        check_diagnostic_with_no_fix(content);
    }

    #[test]
    fn bool_no_diagnostic() {
        let content = r"
            fn test_fn() {
                match false {
                    true => {}
                    false => {}
                }
            }
        ";

        check_no_diagnostic(content);
    }

    #[test]
    fn tuple_of_bools_no_arms() {
        let content = r"
            fn test_fn() {
                match (false, true) {
                }
            }
        ";

        check_diagnostic_with_no_fix(content);
    }

    #[test]
    fn tuple_of_bools_missing_arms() {
        let content = r"
            fn test_fn() {
                match (false, true) {
                    (true, true) => {},
                }
            }
        ";

        check_diagnostic_with_no_fix(content);
    }

    #[test]
    fn tuple_of_bools_no_diagnostic() {
        let content = r"
            fn test_fn() {
                match (false, true) {
                    (true, true) => {},
                    (true, false) => {},
                    (false, true) => {},
                    (false, false) => {},
                }
            }
        ";

        check_no_diagnostic(content);
    }

    #[test]
    fn tuple_of_tuple_and_bools_no_arms() {
        let content = r"
            fn test_fn() {
                match (false, ((), false)) {
                }
            }
        ";

        check_diagnostic_with_no_fix(content);
    }

    #[test]
    fn tuple_of_tuple_and_bools_missing_arms() {
        let content = r"
            fn test_fn() {
                match (false, ((), false)) {
                    (true, ((), true)) => {},
                }
            }
        ";

        check_diagnostic_with_no_fix(content);
    }

    #[test]
    fn tuple_of_tuple_and_bools_no_diagnostic() {
        let content = r"
            fn test_fn() {
                match (false, ((), false)) {
                    (true, ((), true)) => {},
                    (true, ((), false)) => {},
                    (false, ((), true)) => {},
                    (false, ((), false)) => {},
                }
            }
        ";

        check_no_diagnostic(content);
    }

    #[test]
    fn tuple_of_tuple_and_bools_wildcard_missing_arms() {
        let content = r"
            fn test_fn() {
                match (false, ((), false)) {
                    (true, _) => {},
                }
            }
        ";

        check_diagnostic_with_no_fix(content);
    }

    #[test]
    fn tuple_of_tuple_and_bools_wildcard_no_diagnostic() {
        let content = r"
            fn test_fn() {
                match (false, ((), false)) {
                    (true, ((), true)) => {},
                    (true, ((), false)) => {},
                    (false, _) => {},
                }
            }
        ";

        check_no_diagnostic(content);
    }

    #[test]
    fn enum_no_arms() {
        let content = r"
            enum Either {
                A,
                B,
            }
            fn test_fn() {
                match Either::A {
                }
            }
        ";

        check_diagnostic_with_no_fix(content);
    }

    #[test]
    fn enum_missing_arms() {
        let content = r"
            enum Either {
                A,
                B,
            }
            fn test_fn() {
                match Either::B {
                    Either::A => {},
                }
            }
        ";

        check_diagnostic_with_no_fix(content);
    }

    #[test]
    fn enum_no_diagnostic() {
        let content = r"
            enum Either {
                A,
                B,
            }
            fn test_fn() {
                match Either::B {
                    Either::A => {},
                    Either::B => {},
                }
            }
        ";

        check_no_diagnostic(content);
    }

    #[test]
    fn enum_containing_bool_no_arms() {
        let content = r"
            enum Either {
                A(bool),
                B,
            }
            fn test_fn() {
                match Either::B {
                }
            }
        ";

        check_diagnostic_with_no_fix(content);
    }

    #[test]
    fn enum_containing_bool_missing_arms() {
        let content = r"
            enum Either {
                A(bool),
                B,
            }
            fn test_fn() {
                match Either::B {
                    Either::A(true) => (),
                    Either::B => (),
                }
            }
        ";

        check_diagnostic_with_no_fix(content);
    }

    #[test]
    fn enum_containing_bool_no_diagnostic() {
        let content = r"
            enum Either {
                A(bool),
                B,
            }
            fn test_fn() {
                match Either::B {
                    Either::A(true) => (),
                    Either::A(false) => (),
                    Either::B => (),
                }
            }
        ";

        check_no_diagnostic(content);
    }

    #[test]
    fn enum_containing_bool_with_wild_no_diagnostic() {
        let content = r"
            enum Either {
                A(bool),
                B,
            }
            fn test_fn() {
                match Either::B {
                    Either::B => (),
                    _ => (),
                }
            }
        ";

        check_no_diagnostic(content);
    }

    #[test]
    fn enum_containing_bool_with_wild_2_no_diagnostic() {
        let content = r"
            enum Either {
                A(bool),
                B,
            }
            fn test_fn() {
                match Either::B {
                    Either::A(_) => (),
                    Either::B => (),
                }
            }
        ";

        check_no_diagnostic(content);
    }

    #[test]
    fn enum_different_sizes_missing_arms() {
        let content = r"
            enum Either {
                A(bool),
                B(bool, bool),
            }
            fn test_fn() {
                match Either::A(false) {
                    Either::A(_) => (),
                    Either::B(false, _) => (),
                }
            }
        ";

        check_diagnostic_with_no_fix(content);
    }

    #[test]
    fn enum_different_sizes_no_diagnostic() {
        let content = r"
            enum Either {
                A(bool),
                B(bool, bool),
            }
            fn test_fn() {
                match Either::A(false) {
                    Either::A(_) => (),
                    Either::B(true, _) => (),
                    Either::B(false, _) => (),
                }
            }
        ";

        check_no_diagnostic(content);
    }

    #[test]
    fn or_no_diagnostic() {
        let content = r"
            enum Either {
                A(bool),
                B(bool, bool),
            }
            fn test_fn() {
                match Either::A(false) {
                    Either::A(true) | Either::A(false) => (),
                    Either::B(true, _) => (),
                    Either::B(false, _) => (),
                }
            }
        ";

        check_no_diagnostic(content);
    }

    #[test]
    fn tuple_of_enum_no_diagnostic() {
        let content = r"
            enum Either {
                A(bool),
                B(bool, bool),
            }
            enum Either2 {
                C,
                D,
            }
            fn test_fn() {
                match (Either::A(false), Either2::C) {
                    (Either::A(true), _) | (Either::A(false), _) => (),
                    (Either::B(true, _), Either2::C) => (),
                    (Either::B(false, _), Either2::C) => (),
                    (Either::B(_, _), Either2::D) => (),
                }
            }
        ";

        check_no_diagnostic(content);
    }
}

#[cfg(test)]
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

    use super::tests::*;

    #[test]
    fn mismatched_types() {
        let content = r"
            enum Either {
                A,
                B,
            }
            enum Either2 {
                C,
                D,
            }
            fn test_fn() {
                match Either::A {
                    Either2::C => (),
                    Either2::D => (),
                }
            }
        ";

        // This is a false negative.
        // We don't currently check that the match arms actually
        // match the type of the match expression.
        check_no_diagnostic(content);
    }
}
