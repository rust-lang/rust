//! Structural Search Replace
//!
//! Allows searching the AST for code that matches one or more patterns and then replacing that code
//! based on a template.

mod matching;
mod nester;
mod parsing;
mod replacing;
mod resolving;
mod search;
#[macro_use]
mod errors;
#[cfg(test)]
mod tests;

use crate::errors::bail;
pub use crate::errors::SsrError;
pub use crate::matching::Match;
use crate::matching::MatchFailureReason;
use hir::Semantics;
use ra_db::{FileId, FilePosition, FileRange};
use ra_ide_db::source_change::SourceFileEdit;
use ra_syntax::{ast, AstNode, SyntaxNode, TextRange};
use resolving::ResolvedRule;
use rustc_hash::FxHashMap;

// A structured search replace rule. Create by calling `parse` on a str.
#[derive(Debug)]
pub struct SsrRule {
    /// A structured pattern that we're searching for.
    pattern: parsing::RawPattern,
    /// What we'll replace it with.
    template: parsing::RawPattern,
    parsed_rules: Vec<parsing::ParsedRule>,
}

#[derive(Debug)]
pub struct SsrPattern {
    raw: parsing::RawPattern,
    parsed_rules: Vec<parsing::ParsedRule>,
}

#[derive(Debug, Default)]
pub struct SsrMatches {
    pub matches: Vec<Match>,
}

/// Searches a crate for pattern matches and possibly replaces them with something else.
pub struct MatchFinder<'db> {
    /// Our source of information about the user's code.
    sema: Semantics<'db, ra_ide_db::RootDatabase>,
    rules: Vec<ResolvedRule>,
    scope: hir::SemanticsScope<'db>,
    hygiene: hir::Hygiene,
}

impl<'db> MatchFinder<'db> {
    /// Constructs a new instance where names will be looked up as if they appeared at
    /// `lookup_context`.
    pub fn in_context(
        db: &'db ra_ide_db::RootDatabase,
        lookup_context: FilePosition,
    ) -> MatchFinder<'db> {
        let sema = Semantics::new(db);
        let file = sema.parse(lookup_context.file_id);
        // Find a node at the requested position, falling back to the whole file.
        let node = file
            .syntax()
            .token_at_offset(lookup_context.offset)
            .left_biased()
            .map(|token| token.parent())
            .unwrap_or_else(|| file.syntax().clone());
        let scope = sema.scope(&node);
        MatchFinder {
            sema: Semantics::new(db),
            rules: Vec::new(),
            scope,
            hygiene: hir::Hygiene::new(db, lookup_context.file_id.into()),
        }
    }

    /// Constructs an instance using the start of the first file in `db` as the lookup context.
    pub fn at_first_file(db: &'db ra_ide_db::RootDatabase) -> Result<MatchFinder<'db>, SsrError> {
        use ra_db::SourceDatabaseExt;
        use ra_ide_db::symbol_index::SymbolsDatabase;
        if let Some(first_file_id) = db
            .local_roots()
            .iter()
            .next()
            .and_then(|root| db.source_root(root.clone()).iter().next())
        {
            Ok(MatchFinder::in_context(
                db,
                FilePosition { file_id: first_file_id, offset: 0.into() },
            ))
        } else {
            bail!("No files to search");
        }
    }

    /// Adds a rule to be applied. The order in which rules are added matters. Earlier rules take
    /// precedence. If a node is matched by an earlier rule, then later rules won't be permitted to
    /// match to it.
    pub fn add_rule(&mut self, rule: SsrRule) -> Result<(), SsrError> {
        for parsed_rule in rule.parsed_rules {
            self.rules.push(ResolvedRule::new(
                parsed_rule,
                &self.scope,
                &self.hygiene,
                self.rules.len(),
            )?);
        }
        Ok(())
    }

    /// Finds matches for all added rules and returns edits for all found matches.
    pub fn edits(&self) -> Vec<SourceFileEdit> {
        use ra_db::SourceDatabaseExt;
        let mut matches_by_file = FxHashMap::default();
        for m in self.matches().matches {
            matches_by_file
                .entry(m.range.file_id)
                .or_insert_with(|| SsrMatches::default())
                .matches
                .push(m);
        }
        let mut edits = vec![];
        for (file_id, matches) in matches_by_file {
            let edit =
                replacing::matches_to_edit(&matches, &self.sema.db.file_text(file_id), &self.rules);
            edits.push(SourceFileEdit { file_id, edit });
        }
        edits
    }

    /// Adds a search pattern. For use if you intend to only call `find_matches_in_file`. If you
    /// intend to do replacement, use `add_rule` instead.
    pub fn add_search_pattern(&mut self, pattern: SsrPattern) -> Result<(), SsrError> {
        for parsed_rule in pattern.parsed_rules {
            self.rules.push(ResolvedRule::new(
                parsed_rule,
                &self.scope,
                &self.hygiene,
                self.rules.len(),
            )?);
        }
        Ok(())
    }

    /// Returns matches for all added rules.
    pub fn matches(&self) -> SsrMatches {
        let mut matches = Vec::new();
        for rule in &self.rules {
            self.find_matches_for_rule(rule, &mut matches);
        }
        nester::nest_and_remove_collisions(matches, &self.sema)
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
                    // For now we ignore rules that have a different kind than our node, otherwise
                    // we get lots of noise. If at some point we add support for restricting rules
                    // to a particular kind of thing (e.g. only match type references), then we can
                    // relax this.
                    if rule.pattern.node.kind() != node.kind() {
                        continue;
                    }
                    out.push(MatchDebugInfo {
                        matched: matching::get_match(true, rule, &node, restrict_range, &self.sema)
                            .map_err(|e| MatchFailureReason {
                                reason: e.reason.unwrap_or_else(|| {
                                    "Match failed, but no reason was given".to_owned()
                                }),
                            }),
                        pattern: rule.pattern.node.clone(),
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
    /// Our search pattern parsed as an expression or item, etc
    pattern: SyntaxNode,
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
        writeln!(f, "{:#?}", self.pattern)?;
        writeln!(f, "============================")?;
        Ok(())
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
