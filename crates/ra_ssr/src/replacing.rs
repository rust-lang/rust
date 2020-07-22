//! Code for applying replacement templates for matches that have previously been found.

use crate::matching::Var;
use crate::{parsing::ParsedRule, Match, SsrMatches};
use ra_syntax::ast::AstToken;
use ra_syntax::{SyntaxElement, SyntaxNode, SyntaxToken, TextSize};
use ra_text_edit::TextEdit;

/// Returns a text edit that will replace each match in `matches` with its corresponding replacement
/// template. Placeholders in the template will have been substituted with whatever they matched to
/// in the original code.
pub(crate) fn matches_to_edit(
    matches: &SsrMatches,
    file_src: &str,
    rules: &[ParsedRule],
) -> TextEdit {
    matches_to_edit_at_offset(matches, file_src, 0.into(), rules)
}

fn matches_to_edit_at_offset(
    matches: &SsrMatches,
    file_src: &str,
    relative_start: TextSize,
    rules: &[ParsedRule],
) -> TextEdit {
    let mut edit_builder = ra_text_edit::TextEditBuilder::default();
    for m in &matches.matches {
        edit_builder.replace(
            m.range.range.checked_sub(relative_start).unwrap(),
            render_replace(m, file_src, rules),
        );
    }
    edit_builder.finish()
}

struct ReplacementRenderer<'a> {
    match_info: &'a Match,
    file_src: &'a str,
    rules: &'a [ParsedRule],
    rule: &'a ParsedRule,
}

fn render_replace(match_info: &Match, file_src: &str, rules: &[ParsedRule]) -> String {
    let mut out = String::new();
    let rule = &rules[match_info.rule_index];
    let template = rule
        .template
        .as_ref()
        .expect("You called MatchFinder::edits after calling MatchFinder::add_search_pattern");
    let renderer = ReplacementRenderer { match_info, file_src, rules, rule };
    renderer.render_node_children(&template, &mut out);
    for comment in &match_info.ignored_comments {
        out.push_str(&comment.syntax().to_string());
    }
    out
}

impl ReplacementRenderer<'_> {
    fn render_node_children(&self, node: &SyntaxNode, out: &mut String) {
        for node_or_token in node.children_with_tokens() {
            self.render_node_or_token(&node_or_token, out);
        }
    }

    fn render_node_or_token(&self, node_or_token: &SyntaxElement, out: &mut String) {
        match node_or_token {
            SyntaxElement::Token(token) => {
                self.render_token(&token, out);
            }
            SyntaxElement::Node(child_node) => {
                self.render_node_children(&child_node, out);
            }
        }
    }

    fn render_token(&self, token: &SyntaxToken, out: &mut String) {
        if let Some(placeholder) = self.rule.get_placeholder(&token) {
            if let Some(placeholder_value) =
                self.match_info.placeholder_values.get(&Var(placeholder.ident.to_string()))
            {
                let range = &placeholder_value.range.range;
                let mut matched_text =
                    self.file_src[usize::from(range.start())..usize::from(range.end())].to_owned();
                let edit = matches_to_edit_at_offset(
                    &placeholder_value.inner_matches,
                    self.file_src,
                    range.start(),
                    self.rules,
                );
                edit.apply(&mut matched_text);
                out.push_str(&matched_text);
            } else {
                // We validated that all placeholder references were valid before we
                // started, so this shouldn't happen.
                panic!(
                    "Internal error: replacement referenced unknown placeholder {}",
                    placeholder.ident
                );
            }
        } else {
            out.push_str(token.text().as_str());
        }
    }
}
