//! Structural Search Replace
//!
//! Allows searching the AST for code that matches one or more patterns and then replacing that code
//! based on a template.

mod matching;
mod parsing;
mod replacing;
#[cfg(test)]
mod tests;

use crate::matching::Match;
use hir::Semantics;
use ra_db::{FileId, FileRange};
use ra_syntax::{ast, AstNode, SmolStr, SyntaxNode};
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
struct SsrPattern {
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

#[derive(Debug, PartialEq)]
pub struct SsrError(String);

#[derive(Debug, Default)]
pub struct SsrMatches {
    matches: Vec<Match>,
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

    pub fn edits_for_file(&self, file_id: FileId) -> Option<TextEdit> {
        let matches = self.find_matches_in_file(file_id);
        if matches.matches.is_empty() {
            None
        } else {
            Some(replacing::matches_to_edit(&matches))
        }
    }

    fn find_matches_in_file(&self, file_id: FileId) -> SsrMatches {
        let file = self.sema.parse(file_id);
        let code = file.syntax();
        let mut matches = SsrMatches::default();
        self.find_matches(code, &None, &mut matches);
        matches
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
}

impl std::fmt::Display for SsrError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Parse error: {}", self.0)
    }
}

impl std::error::Error for SsrError {}
