//!  structural search replace

use crate::source_change::SourceFileEdit;
use ra_ide_db::RootDatabase;
use ra_syntax::ast::make::expr_from_text;
use ra_syntax::ast::{AstToken, Comment};
use ra_syntax::{AstNode, SyntaxElement, SyntaxNode};
use ra_text_edit::{TextEdit, TextEditBuilder};
use rustc_hash::FxHashMap;
use std::collections::HashMap;
use std::str::FromStr;

pub use ra_db::{SourceDatabase, SourceDatabaseExt};
use ra_ide_db::symbol_index::SymbolsDatabase;

#[derive(Debug, PartialEq)]
pub struct SsrError(String);

impl std::fmt::Display for SsrError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Parse error: {}", self.0)
    }
}

impl std::error::Error for SsrError {}

pub fn parse_search_replace(
    query: &str,
    db: &RootDatabase,
) -> Result<Vec<SourceFileEdit>, SsrError> {
    let mut edits = vec![];
    let query: SsrQuery = query.parse()?;
    for &root in db.local_roots().iter() {
        let sr = db.source_root(root);
        for file_id in sr.walk() {
            dbg!(db.file_relative_path(file_id));
            let matches = find(&query.pattern, db.parse(file_id).tree().syntax());
            if !matches.matches.is_empty() {
                edits.push(SourceFileEdit { file_id, edit: replace(&matches, &query.template) });
            }
        }
    }
    Ok(edits)
}

#[derive(Debug)]
struct SsrQuery {
    pattern: SsrPattern,
    template: SsrTemplate,
}

#[derive(Debug)]
struct SsrPattern {
    pattern: SyntaxNode,
    vars: Vec<Var>,
}

/// represents an `$var` in an SSR query
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Var(String);

#[derive(Debug)]
struct SsrTemplate {
    template: SyntaxNode,
    placeholders: FxHashMap<SyntaxNode, Var>,
}

type Binding = HashMap<Var, SyntaxNode>;

#[derive(Debug)]
struct Match {
    place: SyntaxNode,
    binding: Binding,
    ignored_comments: Vec<Comment>,
}

#[derive(Debug)]
struct SsrMatches {
    matches: Vec<Match>,
}

impl FromStr for SsrQuery {
    type Err = SsrError;

    fn from_str(query: &str) -> Result<SsrQuery, SsrError> {
        let mut it = query.split("==>>");
        let pattern = it.next().expect("at least empty string").trim();
        let mut template = it
            .next()
            .ok_or_else(|| SsrError("Cannot find delemiter `==>>`".into()))?
            .trim()
            .to_string();
        if it.next().is_some() {
            return Err(SsrError("More than one delimiter found".into()));
        }
        let mut vars = vec![];
        let mut it = pattern.split('$');
        let mut pattern = it.next().expect("something").to_string();

        for part in it.map(split_by_var) {
            let (var, var_type, remainder) = part?;
            is_expr(var_type)?;
            let new_var = create_name(var, &mut vars)?;
            pattern.push_str(new_var);
            pattern.push_str(remainder);
            template = replace_in_template(template, var, new_var);
        }

        let template = expr_from_text(&template).syntax().clone();
        let mut placeholders = FxHashMap::default();

        traverse(&template, &mut |n| {
            if let Some(v) = vars.iter().find(|v| v.0.as_str() == n.text()) {
                placeholders.insert(n.clone(), v.clone());
                false
            } else {
                true
            }
        });

        let pattern = SsrPattern { pattern: expr_from_text(&pattern).syntax().clone(), vars };
        let template = SsrTemplate { template, placeholders };
        Ok(SsrQuery { pattern, template })
    }
}

fn traverse(node: &SyntaxNode, go: &mut impl FnMut(&SyntaxNode) -> bool) {
    if !go(node) {
        return;
    }
    for ref child in node.children() {
        traverse(child, go);
    }
}

fn split_by_var(s: &str) -> Result<(&str, &str, &str), SsrError> {
    let end_of_name = s.find(':').ok_or_else(|| SsrError("Use $<name>:expr".into()))?;
    let name = &s[0..end_of_name];
    is_name(name)?;
    let type_begin = end_of_name + 1;
    let type_length =
        s[type_begin..].find(|c| !char::is_ascii_alphanumeric(&c)).unwrap_or_else(|| s.len());
    let type_name = &s[type_begin..type_begin + type_length];
    Ok((name, type_name, &s[type_begin + type_length..]))
}

fn is_name(s: &str) -> Result<(), SsrError> {
    if s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        Ok(())
    } else {
        Err(SsrError("Name can contain only alphanumerics and _".into()))
    }
}

fn is_expr(s: &str) -> Result<(), SsrError> {
    if s == "expr" {
        Ok(())
    } else {
        Err(SsrError("Only $<name>:expr is supported".into()))
    }
}

fn replace_in_template(template: String, var: &str, new_var: &str) -> String {
    let name = format!("${}", var);
    template.replace(&name, new_var)
}

fn create_name<'a>(name: &str, vars: &'a mut Vec<Var>) -> Result<&'a str, SsrError> {
    let sanitized_name = format!("__search_pattern_{}", name);
    if vars.iter().any(|a| a.0 == sanitized_name) {
        return Err(SsrError(format!("Name `{}` repeats more than once", name)));
    }
    vars.push(Var(sanitized_name));
    Ok(&vars.last().unwrap().0)
}

fn find(pattern: &SsrPattern, code: &SyntaxNode) -> SsrMatches {
    fn check(
        pattern: &SyntaxElement,
        code: &SyntaxElement,
        placeholders: &[Var],
        mut match_: Match,
    ) -> Option<Match> {
        match (pattern, code) {
            (SyntaxElement::Token(ref pattern), SyntaxElement::Token(ref code)) => {
                if pattern.text() == code.text() {
                    Some(match_)
                } else {
                    None
                }
            }
            (SyntaxElement::Node(ref pattern), SyntaxElement::Node(ref code)) => {
                if placeholders.iter().any(|n| n.0.as_str() == pattern.text()) {
                    match_.binding.insert(Var(pattern.text().to_string()), code.clone());
                    Some(match_)
                } else {
                    let mut pattern_children = pattern
                        .children_with_tokens()
                        .filter(|element| !element.kind().is_trivia());
                    let mut code_children =
                        code.children_with_tokens().filter(|element| !element.kind().is_trivia());
                    let new_ignored_comments = code.children_with_tokens().filter_map(|element| {
                        element.as_token().and_then(|token| Comment::cast(token.clone()))
                    });
                    match_.ignored_comments.extend(new_ignored_comments);
                    let match_from_children = pattern_children
                        .by_ref()
                        .zip(code_children.by_ref())
                        .fold(Some(match_), |accum, (a, b)| {
                            accum.and_then(|match_| check(&a, &b, placeholders, match_))
                        });
                    match_from_children.and_then(|match_| {
                        if pattern_children.count() == 0 && code_children.count() == 0 {
                            Some(match_)
                        } else {
                            None
                        }
                    })
                }
            }
            _ => None,
        }
    }
    let kind = pattern.pattern.kind();
    let matches = code
        .descendants_with_tokens()
        .filter(|n| n.kind() == kind)
        .filter_map(|code| {
            let match_ = Match {
                place: code.as_node().unwrap().clone(),
                binding: HashMap::new(),
                ignored_comments: vec![],
            };
            check(&SyntaxElement::from(pattern.pattern.clone()), &code, &pattern.vars, match_)
        })
        .collect();
    SsrMatches { matches }
}

fn replace(matches: &SsrMatches, template: &SsrTemplate) -> TextEdit {
    let mut builder = TextEditBuilder::default();
    for match_ in &matches.matches {
        builder.replace(
            match_.place.text_range(),
            render_replace(&match_.binding, &match_.ignored_comments, template),
        );
    }
    builder.finish()
}

fn render_replace(
    binding: &Binding,
    ignored_comments: &Vec<Comment>,
    template: &SsrTemplate,
) -> String {
    let mut builder = TextEditBuilder::default();
    for element in template.template.descendants() {
        if let Some(var) = template.placeholders.get(&element) {
            builder.replace(element.text_range(), binding[var].to_string())
        }
    }
    for comment in ignored_comments {
        builder.insert(template.template.text_range().end(), comment.syntax().to_string())
    }
    builder.finish().apply(&template.template.text().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ra_syntax::SourceFile;

    fn parse_error_text(query: &str) -> String {
        format!("{}", query.parse::<SsrQuery>().unwrap_err())
    }

    #[test]
    fn parser_happy_case() {
        let result: SsrQuery = "foo($a:expr, $b:expr) ==>> bar($b, $a)".parse().unwrap();
        assert_eq!(&result.pattern.pattern.text(), "foo(__search_pattern_a, __search_pattern_b)");
        assert_eq!(result.pattern.vars.len(), 2);
        assert_eq!(result.pattern.vars[0].0, "__search_pattern_a");
        assert_eq!(result.pattern.vars[1].0, "__search_pattern_b");
        assert_eq!(&result.template.template.text(), "bar(__search_pattern_b, __search_pattern_a)");
        dbg!(result.template.placeholders);
    }

    #[test]
    fn parser_empty_query() {
        assert_eq!(parse_error_text(""), "Parse error: Cannot find delemiter `==>>`");
    }

    #[test]
    fn parser_no_delimiter() {
        assert_eq!(parse_error_text("foo()"), "Parse error: Cannot find delemiter `==>>`");
    }

    #[test]
    fn parser_two_delimiters() {
        assert_eq!(
            parse_error_text("foo() ==>> a ==>> b "),
            "Parse error: More than one delimiter found"
        );
    }

    #[test]
    fn parser_no_pattern_type() {
        assert_eq!(parse_error_text("foo($a) ==>>"), "Parse error: Use $<name>:expr");
    }

    #[test]
    fn parser_invalid_name() {
        assert_eq!(
            parse_error_text("foo($a+:expr) ==>>"),
            "Parse error: Name can contain only alphanumerics and _"
        );
    }

    #[test]
    fn parser_invalid_type() {
        assert_eq!(
            parse_error_text("foo($a:ident) ==>>"),
            "Parse error: Only $<name>:expr is supported"
        );
    }

    #[test]
    fn parser_repeated_name() {
        assert_eq!(
            parse_error_text("foo($a:expr, $a:expr) ==>>"),
            "Parse error: Name `a` repeats more than once"
        );
    }

    #[test]
    fn parse_match_replace() {
        let query: SsrQuery = "foo($x:expr) ==>> bar($x)".parse().unwrap();
        let input = "fn main() { foo(1+2); }";

        let code = SourceFile::parse(input).tree();
        let matches = find(&query.pattern, code.syntax());
        assert_eq!(matches.matches.len(), 1);
        assert_eq!(matches.matches[0].place.text(), "foo(1+2)");
        assert_eq!(matches.matches[0].binding.len(), 1);
        assert_eq!(
            matches.matches[0].binding[&Var("__search_pattern_x".to_string())].text(),
            "1+2"
        );

        let edit = replace(&matches, &query.template);
        assert_eq!(edit.apply(input), "fn main() { bar(1+2); }");
    }

    fn assert_ssr_transform(query: &str, input: &str, result: &str) {
        let query: SsrQuery = query.parse().unwrap();
        let code = SourceFile::parse(input).tree();
        let matches = find(&query.pattern, code.syntax());
        let edit = replace(&matches, &query.template);
        assert_eq!(edit.apply(input), result);
    }

    #[test]
    fn ssr_function_to_method() {
        assert_ssr_transform(
            "my_function($a:expr, $b:expr) ==>> ($a).my_method($b)",
            "loop { my_function( other_func(x, y), z + w) }",
            "loop { (other_func(x, y)).my_method(z + w) }",
        )
    }

    #[test]
    fn ssr_nested_function() {
        assert_ssr_transform(
            "foo($a:expr, $b:expr, $c:expr) ==>> bar($c, baz($a, $b))",
            "fn main { foo  (x + value.method(b), x+y-z, true && false) }",
            "fn main { bar(true && false, baz(x + value.method(b), x+y-z)) }",
        )
    }

    #[test]
    fn ssr_expected_spacing() {
        assert_ssr_transform(
            "foo($x:expr) + bar() ==>> bar($x)",
            "fn main() { foo(5) + bar() }",
            "fn main() { bar(5) }",
        );
    }

    #[test]
    fn ssr_with_extra_space() {
        assert_ssr_transform(
            "foo($x:expr  ) +    bar() ==>> bar($x)",
            "fn main() { foo(  5 )  +bar(   ) }",
            "fn main() { bar(5) }",
        );
    }

    #[test]
    fn ssr_keeps_nested_comment() {
        assert_ssr_transform(
            "foo($x:expr) ==>> bar($x)",
            "fn main() { foo(other(5 /* using 5 */)) }",
            "fn main() { bar(other(5 /* using 5 */)) }",
        )
    }

    #[test]
    fn ssr_keeps_comment() {
        assert_ssr_transform(
            "foo($x:expr) ==>> bar($x)",
            "fn main() { foo(5 /* using 5 */) }",
            "fn main() { bar(5)/* using 5 */ }",
        )
    }
}
