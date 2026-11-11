//! Evaluates type predicates at a given position in the source code.

use hir::{PredicateEvaluationResult, Semantics};
use ide_db::{FilePosition, RootDatabase};
use syntax::{AstNode, SourceFile, ast};

pub(crate) fn evaluate_predicate(
    db: &RootDatabase,
    text: String,
    position: FilePosition,
) -> PredicateEvaluationResult {
    let sema = Semantics::new(db);
    let source_file = sema.parse_guess_edition(position.file_id);
    let edition = sema.attach_first_edition(position.file_id).edition(db);

    let Some(where_clause) = parse_where_clause(&text, edition) else {
        return PredicateEvaluationResult::invalid("expected a single where-clause predicate");
    };

    let node = source_file
        .syntax()
        .token_at_offset(position.offset)
        .next()
        .and_then(|token| token.parent())
        .unwrap_or_else(|| source_file.syntax().clone());
    sema.evaluate_where_clause_at(&node, position.offset, where_clause)
}

fn parse_where_clause(text: &str, edition: span::Edition) -> Option<ast::WhereClause> {
    let text = text.trim().trim_end_matches(',').trim_end();
    let wrapped = format!("fn __ra_evaluate_predicate() where {text}, {{}}");
    let parse = SourceFile::parse(&wrapped, edition);
    if !parse.errors().is_empty() {
        return None;
    }

    let where_clause = parse.tree().syntax().descendants().find_map(ast::WhereClause::cast)?;
    if where_clause.predicates().count() == 1 { Some(where_clause) } else { None }
}

#[cfg(test)]
mod tests {
    use hir::PredicateEvaluationStatus;

    use crate::fixture;

    fn check(ra_fixture: &str, predicate: &str, status: PredicateEvaluationStatus) {
        let (analysis, position) = fixture::position(ra_fixture);
        let result = analysis.evaluate_predicate(predicate.to_owned(), position).unwrap();
        assert_eq!(result.status, status, "{}", result.message);
    }

    #[test]
    fn evaluates_concrete_trait_predicate() {
        check(
            r#"
trait Trait {}
struct S;
impl Trait for S {}
fn f() { $0 }
"#,
            "S: Trait",
            PredicateEvaluationStatus::Holds,
        );
    }

    #[test]
    fn evaluates_generic_bound_from_environment() {
        check(
            r#"
trait Trait {}
fn f<T: Trait>() { $0 }
"#,
            "T: Trait",
            PredicateEvaluationStatus::Holds,
        );
    }

    #[test]
    fn reports_missing_generic_bound_as_not_proven() {
        check(
            r#"
trait Trait {}
fn f<T>() { $0 }
"#,
            "T: Trait",
            PredicateEvaluationStatus::NotProven,
        );
    }

    #[test]
    fn evaluates_associated_type_binding() {
        check(
            r#"
trait Iterator { type Item; }
fn f<I: Iterator<Item = u32>>() { $0 }
"#,
            "I: Iterator<Item = u32>",
            PredicateEvaluationStatus::Holds,
        );
    }

    #[test]
    fn reports_unresolved_type_as_invalid() {
        check(
            r#"
trait Trait {}
fn f() { $0 }
"#,
            "Type: Trait",
            PredicateEvaluationStatus::Invalid,
        );
    }

    #[test]
    fn reports_unresolved_trait_as_invalid() {
        check(
            r#"
struct Type;
fn f() { $0 }
"#,
            "Type: Trait",
            PredicateEvaluationStatus::Invalid,
        );
    }

    #[test]
    fn evaluates_lifetime_predicate() {
        check(
            r#"
fn f<'a, 'b>()
where
    'a: 'b,
{
    $0
}
"#,
            "'a: 'b",
            PredicateEvaluationStatus::Holds,
        );
    }

    #[test]
    fn evaluates_type_outlives_predicate() {
        check(
            r#"
fn f<T: 'static>() { $0 }
"#,
            "T: 'static",
            PredicateEvaluationStatus::Holds,
        );
    }

    #[test]
    fn rejects_invalid_predicate() {
        check(
            r#"
trait Trait {}
fn f() { $0 }
"#,
            "u32 Trait",
            PredicateEvaluationStatus::Invalid,
        );
    }
}
