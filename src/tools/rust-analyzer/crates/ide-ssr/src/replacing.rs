//! Code for applying replacement templates for matches that have previously been found.

use ide_db::{FxHashMap, FxHashSet};
use itertools::Itertools;
use syntax::{
    ast::{self, AstNode, AstToken},
    SyntaxElement, SyntaxKind, SyntaxNode, SyntaxToken, TextRange, TextSize,
};
use text_edit::TextEdit;

use crate::{fragments, resolving::ResolvedRule, Match, SsrMatches};

/// Returns a text edit that will replace each match in `matches` with its corresponding replacement
/// template. Placeholders in the template will have been substituted with whatever they matched to
/// in the original code.
pub(crate) fn matches_to_edit(
    db: &dyn hir::db::ExpandDatabase,
    matches: &SsrMatches,
    file_src: &str,
    rules: &[ResolvedRule],
) -> TextEdit {
    matches_to_edit_at_offset(db, matches, file_src, 0.into(), rules)
}

fn matches_to_edit_at_offset(
    db: &dyn hir::db::ExpandDatabase,
    matches: &SsrMatches,
    file_src: &str,
    relative_start: TextSize,
    rules: &[ResolvedRule],
) -> TextEdit {
    let mut edit_builder = TextEdit::builder();
    for m in &matches.matches {
        edit_builder.replace(
            m.range.range.checked_sub(relative_start).unwrap(),
            render_replace(db, m, file_src, rules),
        );
    }
    edit_builder.finish()
}

struct ReplacementRenderer<'a> {
    db: &'a dyn hir::db::ExpandDatabase,
    match_info: &'a Match,
    file_src: &'a str,
    rules: &'a [ResolvedRule],
    rule: &'a ResolvedRule,
    out: String,
    // Map from a range within `out` to a token in `template` that represents a placeholder. This is
    // used to validate that the generated source code doesn't split any placeholder expansions (see
    // below).
    placeholder_tokens_by_range: FxHashMap<TextRange, SyntaxToken>,
    // Which placeholder tokens need to be wrapped in parenthesis in order to ensure that when `out`
    // is parsed, placeholders don't get split. e.g. if a template of `$a.to_string()` results in `1
    // + 2.to_string()` then the placeholder value `1 + 2` was split and needs parenthesis.
    placeholder_tokens_requiring_parenthesis: FxHashSet<SyntaxToken>,
}

fn render_replace(
    db: &dyn hir::db::ExpandDatabase,
    match_info: &Match,
    file_src: &str,
    rules: &[ResolvedRule],
) -> String {
    let rule = &rules[match_info.rule_index];
    let template = rule
        .template
        .as_ref()
        .expect("You called MatchFinder::edits after calling MatchFinder::add_search_pattern");
    let mut renderer = ReplacementRenderer {
        db,
        match_info,
        file_src,
        rules,
        rule,
        out: String::new(),
        placeholder_tokens_requiring_parenthesis: FxHashSet::default(),
        placeholder_tokens_by_range: FxHashMap::default(),
    };
    renderer.render_node(&template.node);
    renderer.maybe_rerender_with_extra_parenthesis(&template.node);
    for comment in &match_info.ignored_comments {
        renderer.out.push_str(&comment.syntax().to_string());
    }
    renderer.out
}

impl ReplacementRenderer<'_> {
    fn render_node_children(&mut self, node: &SyntaxNode) {
        for node_or_token in node.children_with_tokens() {
            self.render_node_or_token(&node_or_token);
        }
    }

    fn render_node_or_token(&mut self, node_or_token: &SyntaxElement) {
        match node_or_token {
            SyntaxElement::Token(token) => {
                self.render_token(token);
            }
            SyntaxElement::Node(child_node) => {
                self.render_node(child_node);
            }
        }
    }

    fn render_node(&mut self, node: &SyntaxNode) {
        if let Some(mod_path) = self.match_info.rendered_template_paths.get(node) {
            self.out.push_str(&mod_path.display(self.db).to_string());
            // Emit everything except for the segment's name-ref, since we already effectively
            // emitted that as part of `mod_path`.
            if let Some(path) = ast::Path::cast(node.clone()) {
                if let Some(segment) = path.segment() {
                    for node_or_token in segment.syntax().children_with_tokens() {
                        if node_or_token.kind() != SyntaxKind::NAME_REF {
                            self.render_node_or_token(&node_or_token);
                        }
                    }
                }
            }
        } else {
            self.render_node_children(node);
        }
    }

    fn render_token(&mut self, token: &SyntaxToken) {
        if let Some(placeholder) = self.rule.get_placeholder(token) {
            if let Some(placeholder_value) =
                self.match_info.placeholder_values.get(&placeholder.ident)
            {
                let range = &placeholder_value.range.range;
                let mut matched_text =
                    self.file_src[usize::from(range.start())..usize::from(range.end())].to_owned();
                // If a method call is performed directly on the placeholder, then autoderef and
                // autoref will apply, so we can just substitute whatever the placeholder matched to
                // directly. If we're not applying a method call, then we need to add explicitly
                // deref and ref in order to match whatever was being done implicitly at the match
                // site.
                if !token_is_method_call_receiver(token)
                    && (placeholder_value.autoderef_count > 0
                        || placeholder_value.autoref_kind != ast::SelfParamKind::Owned)
                {
                    cov_mark::hit!(replace_autoref_autoderef_capture);
                    let ref_kind = match placeholder_value.autoref_kind {
                        ast::SelfParamKind::Owned => "",
                        ast::SelfParamKind::Ref => "&",
                        ast::SelfParamKind::MutRef => "&mut ",
                    };
                    matched_text = format!(
                        "{}{}{}",
                        ref_kind,
                        "*".repeat(placeholder_value.autoderef_count),
                        matched_text
                    );
                }
                let edit = matches_to_edit_at_offset(
                    self.db,
                    &placeholder_value.inner_matches,
                    self.file_src,
                    range.start(),
                    self.rules,
                );
                let needs_parenthesis =
                    self.placeholder_tokens_requiring_parenthesis.contains(token);
                edit.apply(&mut matched_text);
                if needs_parenthesis {
                    self.out.push('(');
                }
                self.placeholder_tokens_by_range.insert(
                    TextRange::new(
                        TextSize::of(&self.out),
                        TextSize::of(&self.out) + TextSize::of(&matched_text),
                    ),
                    token.clone(),
                );
                self.out.push_str(&matched_text);
                if needs_parenthesis {
                    self.out.push(')');
                }
            } else {
                // We validated that all placeholder references were valid before we
                // started, so this shouldn't happen.
                panic!(
                    "Internal error: replacement referenced unknown placeholder {}",
                    placeholder.ident
                );
            }
        } else {
            self.out.push_str(token.text());
        }
    }

    // Checks if the resulting code, when parsed doesn't split any placeholders due to different
    // order of operations between the search pattern and the replacement template. If any do, then
    // we rerender the template and wrap the problematic placeholders with parenthesis.
    fn maybe_rerender_with_extra_parenthesis(&mut self, template: &SyntaxNode) {
        if let Some(node) = parse_as_kind(&self.out, template.kind()) {
            self.remove_node_ranges(node);
            if self.placeholder_tokens_by_range.is_empty() {
                return;
            }
            self.placeholder_tokens_requiring_parenthesis =
                self.placeholder_tokens_by_range.values().cloned().collect();
            self.out.clear();
            self.render_node(template);
        }
    }

    fn remove_node_ranges(&mut self, node: SyntaxNode) {
        self.placeholder_tokens_by_range.remove(&node.text_range());
        for child in node.children() {
            self.remove_node_ranges(child);
        }
    }
}

/// Returns whether token is the receiver of a method call. Note, being within the receiver of a
/// method call doesn't count. e.g. if the token is `$a`, then `$a.foo()` will return true, while
/// `($a + $b).foo()` or `x.foo($a)` will return false.
fn token_is_method_call_receiver(token: &SyntaxToken) -> bool {
    // Find the first method call among the ancestors of `token`, then check if the only token
    // within the receiver is `token`.
    if let Some(receiver) = token
        .parent_ancestors()
        .find_map(ast::MethodCallExpr::cast)
        .and_then(|call| call.receiver())
    {
        let tokens = receiver.syntax().descendants_with_tokens().filter_map(|node_or_token| {
            match node_or_token {
                SyntaxElement::Token(t) => Some(t),
                _ => None,
            }
        });
        if let Some((only_token,)) = tokens.collect_tuple() {
            return only_token == *token;
        }
    }
    false
}

fn parse_as_kind(code: &str, kind: SyntaxKind) -> Option<SyntaxNode> {
    if ast::Expr::can_cast(kind) {
        if let Ok(expr) = fragments::expr(code) {
            return Some(expr);
        }
    }
    if ast::Item::can_cast(kind) {
        if let Ok(item) = fragments::item(code) {
            return Some(item);
        }
    }
    None
}
