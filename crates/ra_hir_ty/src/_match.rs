//! This module implements match statement exhaustiveness checking and usefulness checking
//! for match arms.
//!
//! It is modeled on the rustc module `librustc_mir_build::hair::pattern::_match`, which
//! contains very detailed documentation about the algorithms used here.
use std::sync::Arc;

use smallvec::{smallvec, SmallVec};

use crate::{
    db::HirDatabase,
    expr::{Body, Expr, Literal, Pat, PatId},
    InferenceResult,
};
use hir_def::{adt::VariantData, EnumVariantId, VariantId};

#[derive(Debug, Clone, Copy)]
/// Either a pattern from the source code being analyzed, represented as
/// as `PatId`, or a `Wild` pattern which is created as an intermediate
/// step in the match checking algorithm and thus is not backed by a
/// real `PatId`.
///
/// Note that it is totally valid for the `PatId` variant to contain
/// a `PatId` which resolves to a `Wild` pattern, if that wild pattern
/// exists in the source code being analyzed.
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatchCheckNotImplemented;

/// The return type of `is_useful` is either an indication of usefulness
/// of the match arm, or an error in the case the match statement
/// is made up of types for which exhaustiveness checking is currently
/// not completely implemented.
///
/// The `std::result::Result` type is used here rather than a custom enum
/// to allow the use of `?`.
pub type MatchCheckResult<T> = Result<T, MatchCheckNotImplemented>;

#[derive(Debug)]
/// A row in a Matrix.
///
/// This type is modeled from the struct of the same name in `rustc`.
pub(crate) struct PatStack(PatStackInner);
type PatStackInner = SmallVec<[PatIdOrWild; 2]>;

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

    /// Computes `D(self)`.
    ///
    /// See the module docs and the associated documentation in rustc for details.
    fn specialize_wildcard(&self, cx: &MatchCheckCtx) -> Option<PatStack> {
        if matches!(self.head().as_pat(cx), Pat::Wild) {
            Some(self.to_tail())
        } else {
            None
        }
    }

    /// Computes `S(constructor, self)`.
    ///
    /// See the module docs and the associated documentation in rustc for details.
    fn specialize_constructor(
        &self,
        cx: &MatchCheckCtx,
        constructor: &Constructor,
    ) -> MatchCheckResult<Option<PatStack>> {
        let result = match (self.head().as_pat(cx), constructor) {
            (Pat::Tuple(ref pat_ids), Constructor::Tuple { arity }) => {
                if pat_ids.len() != *arity {
                    None
                } else {
                    Some(self.replace_head_with(pat_ids))
                }
            }
            (Pat::Lit(_), Constructor::Bool(_)) => {
                // for now we only support bool literals
                Some(self.to_tail())
            }
            (Pat::Wild, constructor) => Some(self.expand_wildcard(cx, constructor)?),
            (Pat::Path(_), Constructor::Enum(constructor)) => {
                // enums with no associated data become `Pat::Path`
                let pat_id = self.head().as_id().expect("we know this isn't a wild");
                if !enum_variant_matches(cx, pat_id, *constructor) {
                    None
                } else {
                    Some(self.to_tail())
                }
            }
            (Pat::TupleStruct { args: ref pat_ids, .. }, Constructor::Enum(constructor)) => {
                let pat_id = self.head().as_id().expect("we know this isn't a wild");
                if !enum_variant_matches(cx, pat_id, *constructor) {
                    None
                } else {
                    Some(self.replace_head_with(pat_ids))
                }
            }
            (Pat::Or(_), _) => unreachable!("we desugar or patterns so this should never happen"),
            (_, _) => return Err(MatchCheckNotImplemented),
        };

        Ok(result)
    }

    /// A special case of `specialize_constructor` where the head of the pattern stack
    /// is a Wild pattern.
    ///
    /// Replaces the Wild pattern at the head of the pattern stack with N Wild patterns
    /// (N >= 0), where N is the arity of the given constructor.
    fn expand_wildcard(
        &self,
        cx: &MatchCheckCtx,
        constructor: &Constructor,
    ) -> MatchCheckResult<PatStack> {
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
                    _ => return Err(MatchCheckNotImplemented),
                }
            }
        };

        for _ in 0..arity {
            patterns.push(PatIdOrWild::Wild);
        }

        for pat in &self.0[1..] {
            patterns.push(*pat);
        }

        Ok(PatStack::from_vec(patterns))
    }
}

#[derive(Debug)]
/// A collection of PatStack.
///
/// This type is modeled from the struct of the same name in `rustc`.
pub(crate) struct Matrix(Vec<PatStack>);

impl Matrix {
    pub(crate) fn empty() -> Self {
        Self(vec![])
    }

    pub(crate) fn push(&mut self, cx: &MatchCheckCtx, row: PatStack) {
        if let Some(Pat::Or(pat_ids)) = row.get_head().map(|pat_id| pat_id.as_pat(cx)) {
            // Or patterns are expanded here
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

    /// Computes `D(self)` for each contained PatStack.
    ///
    /// See the module docs and the associated documentation in rustc for details.
    fn specialize_wildcard(&self, cx: &MatchCheckCtx) -> Self {
        Self::collect(cx, self.0.iter().filter_map(|r| r.specialize_wildcard(cx)))
    }

    /// Computes `S(constructor, self)` for each contained PatStack.
    ///
    /// See the module docs and the associated documentation in rustc for details.
    fn specialize_constructor(
        &self,
        cx: &MatchCheckCtx,
        constructor: &Constructor,
    ) -> MatchCheckResult<Self> {
        let mut new_matrix = Matrix::empty();
        for pat in &self.0 {
            if let Some(pat) = pat.specialize_constructor(cx, constructor)? {
                new_matrix.push(cx, pat);
            }
        }

        Ok(new_matrix)
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
/// An indication of the usefulness of a given match arm, where
/// usefulness is defined as matching some patterns which were
/// not matched by an prior match arms.
///
/// We may eventually need an `Unknown` variant here.
pub enum Usefulness {
    Useful,
    NotUseful,
}

pub struct MatchCheckCtx<'a> {
    pub body: Arc<Body>,
    pub infer: Arc<InferenceResult>,
    pub db: &'a dyn HirDatabase,
}

/// Given a set of patterns `matrix`, and pattern to consider `v`, determines
/// whether `v` is useful. A pattern is useful if it covers cases which were
/// not previously covered.
///
/// When calling this function externally (that is, not the recursive calls) it
/// expected that you have already type checked the match arms. All patterns in
/// matrix should be the same type as v, as well as they should all be the same
/// type as the match expression.
pub(crate) fn is_useful(
    cx: &MatchCheckCtx,
    matrix: &Matrix,
    v: &PatStack,
) -> MatchCheckResult<Usefulness> {
    if v.is_empty() {
        let result = if matrix.is_empty() { Usefulness::Useful } else { Usefulness::NotUseful };

        return Ok(result);
    }

    if let Pat::Or(pat_ids) = v.head().as_pat(cx) {
        let mut found_unimplemented = false;
        let any_useful = pat_ids.iter().any(|&pat_id| {
            let v = PatStack::from_pattern(pat_id);

            match is_useful(cx, matrix, &v) {
                Ok(Usefulness::Useful) => true,
                Ok(Usefulness::NotUseful) => false,
                _ => {
                    found_unimplemented = true;
                    false
                }
            }
        });

        return if any_useful {
            Ok(Usefulness::Useful)
        } else if found_unimplemented {
            Err(MatchCheckNotImplemented)
        } else {
            Ok(Usefulness::NotUseful)
        };
    }

    if let Some(constructor) = pat_constructor(cx, v.head())? {
        let matrix = matrix.specialize_constructor(&cx, &constructor)?;
        let v = v
            .specialize_constructor(&cx, &constructor)?
            .expect("we know this can't fail because we get the constructor from `v.head()` above");

        is_useful(&cx, &matrix, &v)
    } else {
        // expanding wildcard
        let mut used_constructors: Vec<Constructor> = vec![];
        for pat in matrix.heads() {
            if let Some(constructor) = pat_constructor(cx, pat)? {
                used_constructors.push(constructor);
            }
        }

        // We assume here that the first constructor is the "correct" type. Since we
        // only care about the "type" of the constructor (i.e. if it is a bool we
        // don't care about the value), this assumption should be valid as long as
        // the match statement is well formed. We currently uphold this invariant by
        // filtering match arms before calling `is_useful`, only passing in match arms
        // whose type matches the type of the match expression.
        match &used_constructors.first() {
            Some(constructor) if all_constructors_covered(&cx, constructor, &used_constructors) => {
                // If all constructors are covered, then we need to consider whether
                // any values are covered by this wildcard.
                //
                // For example, with matrix '[[Some(true)], [None]]', all
                // constructors are covered (`Some`/`None`), so we need
                // to perform specialization to see that our wildcard will cover
                // the `Some(false)` case.
                let mut constructor = None;
                for pat in matrix.heads() {
                    if let Some(c) = pat_constructor(cx, pat)? {
                        constructor = Some(c);
                        break;
                    }
                }

                if let Some(constructor) = constructor {
                    if let Constructor::Enum(e) = constructor {
                        // For enums we handle each variant as a distinct constructor, so
                        // here we create a constructor for each variant and then check
                        // usefulness after specializing for that constructor.
                        let mut found_unimplemented = false;
                        for constructor in
                            cx.db.enum_data(e.parent).variants.iter().map(|(local_id, _)| {
                                Constructor::Enum(EnumVariantId { parent: e.parent, local_id })
                            })
                        {
                            let matrix = matrix.specialize_constructor(&cx, &constructor)?;
                            let v = v.expand_wildcard(&cx, &constructor)?;

                            match is_useful(&cx, &matrix, &v) {
                                Ok(Usefulness::Useful) => return Ok(Usefulness::Useful),
                                Ok(Usefulness::NotUseful) => continue,
                                _ => found_unimplemented = true,
                            };
                        }

                        if found_unimplemented {
                            Err(MatchCheckNotImplemented)
                        } else {
                            Ok(Usefulness::NotUseful)
                        }
                    } else {
                        let matrix = matrix.specialize_constructor(&cx, &constructor)?;
                        let v = v.expand_wildcard(&cx, &constructor)?;

                        is_useful(&cx, &matrix, &v)
                    }
                } else {
                    Ok(Usefulness::NotUseful)
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
/// Similar to TypeCtor, but includes additional information about the specific
/// value being instantiated. For example, TypeCtor::Bool doesn't contain the
/// boolean value.
enum Constructor {
    Bool(bool),
    Tuple { arity: usize },
    Enum(EnumVariantId),
}

/// Returns the constructor for the given pattern. Should only return None
/// in the case of a Wild pattern.
fn pat_constructor(cx: &MatchCheckCtx, pat: PatIdOrWild) -> MatchCheckResult<Option<Constructor>> {
    let res = match pat.as_pat(cx) {
        Pat::Wild => None,
        Pat::Tuple(pats) => Some(Constructor::Tuple { arity: pats.len() }),
        Pat::Lit(lit_expr) => match cx.body.exprs[lit_expr] {
            Expr::Literal(Literal::Bool(val)) => Some(Constructor::Bool(val)),
            _ => return Err(MatchCheckNotImplemented),
        },
        Pat::TupleStruct { .. } | Pat::Path(_) => {
            let pat_id = pat.as_id().expect("we already know this pattern is not a wild");
            let variant_id =
                cx.infer.variant_resolution_for_pat(pat_id).ok_or(MatchCheckNotImplemented)?;
            match variant_id {
                VariantId::EnumVariantId(enum_variant_id) => {
                    Some(Constructor::Enum(enum_variant_id))
                }
                _ => return Err(MatchCheckNotImplemented),
            }
        }
        _ => return Err(MatchCheckNotImplemented),
    };

    Ok(res)
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
    fn empty_tuple_wild() {
        let content = r"
            fn test_fn() {
                match () {
                    _ => {}
                }
            }
        ";

        check_no_diagnostic(content);
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
    fn tuple_of_bools_binding_missing_arms() {
        let content = r"
            fn test_fn() {
                match (false, true) {
                    (true, _x) => {},
                }
            }
        ";

        check_diagnostic_with_no_fix(content);
    }

    #[test]
    fn tuple_of_bools_binding_no_diagnostic() {
        let content = r"
            fn test_fn() {
                match (false, true) {
                    (true, _x) => {},
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

        // Match arms with the incorrect type are filtered out.
        check_diagnostic_with_no_fix(content);
    }

    #[test]
    fn mismatched_types_with_different_arity() {
        let content = r"
            fn test_fn() {
                match (true, false) {
                    (true, false, true) => (),
                    (true) => (),
                }
            }
        ";

        // Match arms with the incorrect type are filtered out.
        check_diagnostic_with_no_fix(content);
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
    fn integers() {
        let content = r"
            fn test_fn() {
                match 5 {
                    10 => (),
                    11..20 => (),
                }
            }
        ";

        // This is a false negative.
        // We don't currently check integer exhaustiveness.
        check_no_diagnostic(content);
    }

    #[test]
    fn enum_record() {
        let content = r"
            enum Either {
                A { foo: u32 },
                B,
            }
            fn test_fn() {
                match Either::B {
                    Either::A { foo: 5 } => (),
                }
            }
        ";

        // This is a false negative.
        // We don't currently handle enum record types.
        check_no_diagnostic(content);
    }

    #[test]
    fn enum_not_in_scope() {
        let content = r"
            fn test_fn() {
                match Foo::Bar {
                    Foo::Baz => (),
                }
            }
        ";

        // This is a false negative.
        // The enum is not in scope so we don't perform exhaustiveness checking.
        check_no_diagnostic(content);
    }
}
