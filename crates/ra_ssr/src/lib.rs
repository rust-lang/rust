//! Structural Search Replace
//!
//! Allows searching the AST for code that matches one or more patterns and then replacing that code
//! based on a template.

mod matching;
mod parsing;
mod replacing;
#[macro_use]
mod errors;
#[cfg(test)]
mod tests;

pub use crate::errors::SsrError;
pub use crate::matching::Match;
use crate::matching::{record_match_fails_reasons_scope, MatchFailureReason};
use hir::Semantics;
use ra_db::{FileId, FileRange};
use ra_syntax::{ast, AstNode, SmolStr, SyntaxKind, SyntaxNode, TextRange};
use ra_text_edit::TextEdit;
use rustc_hash::FxHashMap;

// A structured search replace rule. Create by calling `parse` on a str.
#[derive(Debug)]
pub struct SsrRule {
    /// A structured pattern that we're searching for.
    pattern: SsrPattern,
    /// What we'll replace it with.
    template: parsing::SsrTemplate,
}

#[derive(Debug)]
pub struct SsrPattern {
    raw: parsing::RawSearchPattern,
    /// Placeholders keyed by the stand-in ident that we use in Rust source code.
    placeholders_by_stand_in: FxHashMap<SmolStr, parsing::Placeholder>,
    // We store our search pattern, parsed as each different kind of thing we can look for. As we
    // traverse the AST, we get the appropriate one of these for the type of node we're on. For many
    // search patterns, only some of these will be present.
    expr: Option<SyntaxNode>,
    type_ref: Option<SyntaxNode>,
    item: Option<SyntaxNode>,
    path: Option<SyntaxNode>,
    pattern: Option<SyntaxNode>,
}

#[derive(Debug, Default)]
pub struct SsrMatches {
    pub matches: Vec<Match>,
}

/// Searches a crate for pattern matches and possibly replaces them with something else.
pub struct MatchFinder<'db> {
    /// Our source of information about the user's code.
    sema: Semantics<'db, ra_ide_db::RootDatabase>,
    rules: Vec<SsrRule>,
}

impl<'db> MatchFinder<'db> {
    pub fn new(db: &'db ra_ide_db::RootDatabase) -> MatchFinder<'db> {
        MatchFinder { sema: Semantics::new(db), rules: Vec::new() }
    }

    pub fn add_rule(&mut self, rule: SsrRule) {
        self.rules.push(rule);
    }

    /// Adds a search pattern. For use if you intend to only call `find_matches_in_file`. If you
    /// intend to do replacement, use `add_rule` instead.
    pub fn add_search_pattern(&mut self, pattern: SsrPattern) {
        self.add_rule(SsrRule { pattern, template: "()".parse().unwrap() })
    }

    pub fn edits_for_file(&self, file_id: FileId) -> Option<TextEdit> {
        let matches = self.find_matches_in_file(file_id);
        if matches.matches.is_empty() {
            None
        } else {
            use ra_db::SourceDatabaseExt;
            Some(replacing::matches_to_edit(&matches, &self.sema.db.file_text(file_id)))
        }
    }

    pub fn find_matches_in_file(&self, file_id: FileId) -> SsrMatches {
        let file = self.sema.parse(file_id);
        let code = file.syntax();
        let mut matches = SsrMatches::default();
        self.find_matches(code, &None, &mut matches);
        matches
    }

    /// Finds all nodes in `file_id` whose text is exactly equal to `snippet` and attempts to match
    /// them, while recording reasons why they don't match. This API is useful for command
    /// line-based debugging where providing a range is difficult.
    pub fn debug_where_text_equal(&self, file_id: FileId, snippet: &str) -> Vec<MatchDebugInfo> {
        use ra_db::SourceDatabaseExt;
        let file = self.sema.parse(file_id);
        let mut res = Vec::new();
        let file_text = self.sema.db.file_text(file_id);
        let mut remaining_text = file_text.as_str();
        let mut base = 0;
        let len = snippet.len() as u32;
        while let Some(offset) = remaining_text.find(snippet) {
            let start = base + offset as u32;
            let end = start + len;
            self.output_debug_for_nodes_at_range(
                file.syntax(),
                FileRange { file_id, range: TextRange::new(start.into(), end.into()) },
                &None,
                &mut res,
            );
            remaining_text = &remaining_text[offset + snippet.len()..];
            base = end;
        }
        res
    }

    fn find_matches(
        &self,
        code: &SyntaxNode,
        restrict_range: &Option<FileRange>,
        matches_out: &mut SsrMatches,
    ) {
        for rule in &self.rules {
            if let Ok(mut m) = matching::get_match(false, rule, &code, restrict_range, &self.sema) {
                // Continue searching in each of our placeholders.
                for placeholder_value in m.placeholder_values.values_mut() {
                    if let Some(placeholder_node) = &placeholder_value.node {
                        // Don't search our placeholder if it's the entire matched node, otherwise we'd
                        // find the same match over and over until we got a stack overflow.
                        if placeholder_node != code {
                            self.find_matches(
                                placeholder_node,
                                restrict_range,
                                &mut placeholder_value.inner_matches,
                            );
                        }
                    }
                }
                matches_out.matches.push(m);
                return;
            }
        }
        // If we've got a macro call, we already tried matching it pre-expansion, which is the only
        // way to match the whole macro, now try expanding it and matching the expansion.
        if let Some(macro_call) = ast::MacroCall::cast(code.clone()) {
            if let Some(expanded) = self.sema.expand(&macro_call) {
                if let Some(tt) = macro_call.token_tree() {
                    // When matching within a macro expansion, we only want to allow matches of
                    // nodes that originated entirely from within the token tree of the macro call.
                    // i.e. we don't want to match something that came from the macro itself.
                    self.find_matches(
                        &expanded,
                        &Some(self.sema.original_range(tt.syntax())),
                        matches_out,
                    );
                }
            }
        }
        for child in code.children() {
            self.find_matches(&child, restrict_range, matches_out);
        }
    }

    fn output_debug_for_nodes_at_range(
        &self,
        node: &SyntaxNode,
        range: FileRange,
        restrict_range: &Option<FileRange>,
        out: &mut Vec<MatchDebugInfo>,
    ) {
        for node in node.children() {
            let node_range = self.sema.original_range(&node);
            if node_range.file_id != range.file_id || !node_range.range.contains_range(range.range)
            {
                continue;
            }
            if node_range.range == range.range {
                for rule in &self.rules {
                    let pattern =
                        rule.pattern.tree_for_kind_with_reason(node.kind()).map(|p| p.clone());
                    out.push(MatchDebugInfo {
                        matched: matching::get_match(true, rule, &node, restrict_range, &self.sema)
                            .map_err(|e| MatchFailureReason {
                                reason: e.reason.unwrap_or_else(|| {
                                    "Match failed, but no reason was given".to_owned()
                                }),
                            }),
                        pattern,
                        node: node.clone(),
                    });
                }
            } else if let Some(macro_call) = ast::MacroCall::cast(node.clone()) {
                if let Some(expanded) = self.sema.expand(&macro_call) {
                    if let Some(tt) = macro_call.token_tree() {
                        self.output_debug_for_nodes_at_range(
                            &expanded,
                            range,
                            &Some(self.sema.original_range(tt.syntax())),
                            out,
                        );
                    }
                }
            }
            self.output_debug_for_nodes_at_range(&node, range, restrict_range, out);
        }
    }
}

pub struct MatchDebugInfo {
    node: SyntaxNode,
    /// Our search pattern parsed as the same kind of syntax node as `node`. e.g. expression, item,
    /// etc. Will be absent if the pattern can't be parsed as that kind.
    pattern: Result<SyntaxNode, MatchFailureReason>,
    matched: Result<Match, MatchFailureReason>,
}

impl std::fmt::Debug for MatchDebugInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.matched {
            Ok(_) => writeln!(f, "Node matched")?,
            Err(reason) => writeln!(f, "Node failed to match because: {}", reason.reason)?,
        }
        writeln!(
            f,
            "============ AST ===========\n\
            {:#?}",
            self.node
        )?;
        writeln!(f, "========= PATTERN ==========")?;
        match &self.pattern {
            Ok(pattern) => {
                writeln!(f, "{:#?}", pattern)?;
            }
            Err(err) => {
                writeln!(f, "{}", err.reason)?;
            }
        }
        writeln!(f, "============================")?;
        Ok(())
    }
}

impl SsrPattern {
    fn tree_for_kind_with_reason(
        &self,
        kind: SyntaxKind,
    ) -> Result<&SyntaxNode, MatchFailureReason> {
        record_match_fails_reasons_scope(true, || self.tree_for_kind(kind))
            .map_err(|e| MatchFailureReason { reason: e.reason.unwrap() })
    }
}

impl SsrMatches {
    /// Returns `self` with any nested matches removed and made into top-level matches.
    pub fn flattened(self) -> SsrMatches {
        let mut out = SsrMatches::default();
        self.flatten_into(&mut out);
        out
    }

    fn flatten_into(self, out: &mut SsrMatches) {
        for mut m in self.matches {
            for p in m.placeholder_values.values_mut() {
                std::mem::replace(&mut p.inner_matches, SsrMatches::default()).flatten_into(out);
            }
            out.matches.push(m);
        }
    }
}

impl Match {
    pub fn matched_text(&self) -> String {
        self.matched_node.text().to_string()
    }
}

impl std::error::Error for SsrError {}

#[cfg(test)]
impl MatchDebugInfo {
    pub(crate) fn match_failure_reason(&self) -> Option<&str> {
        self.matched.as_ref().err().map(|r| r.reason.as_str())
    }
}
