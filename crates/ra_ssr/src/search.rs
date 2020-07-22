//! Searching for matches.

use crate::{matching, resolving::ResolvedRule, Match, MatchFinder};
use ra_db::FileRange;
use ra_syntax::{ast, AstNode, SyntaxNode};

impl<'db> MatchFinder<'db> {
    /// Adds all matches for `rule` to `matches_out`. Matches may overlap in ways that make
    /// replacement impossible, so further processing is required in order to properly nest matches
    /// and remove overlapping matches. This is done in the `nesting` module.
    pub(crate) fn find_matches_for_rule(&self, rule: &ResolvedRule, matches_out: &mut Vec<Match>) {
        // FIXME: Use resolved paths in the pattern to find places to search instead of always
        // scanning every node.
        self.slow_scan(rule, matches_out);
    }

    fn slow_scan(&self, rule: &ResolvedRule, matches_out: &mut Vec<Match>) {
        use ra_db::SourceDatabaseExt;
        use ra_ide_db::symbol_index::SymbolsDatabase;
        for &root in self.sema.db.local_roots().iter() {
            let sr = self.sema.db.source_root(root);
            for file_id in sr.iter() {
                let file = self.sema.parse(file_id);
                let code = file.syntax();
                self.slow_scan_node(code, rule, &None, matches_out);
            }
        }
    }

    fn slow_scan_node(
        &self,
        code: &SyntaxNode,
        rule: &ResolvedRule,
        restrict_range: &Option<FileRange>,
        matches_out: &mut Vec<Match>,
    ) {
        if let Ok(m) = matching::get_match(false, rule, &code, restrict_range, &self.sema) {
            matches_out.push(m);
        }
        // If we've got a macro call, we already tried matching it pre-expansion, which is the only
        // way to match the whole macro, now try expanding it and matching the expansion.
        if let Some(macro_call) = ast::MacroCall::cast(code.clone()) {
            if let Some(expanded) = self.sema.expand(&macro_call) {
                if let Some(tt) = macro_call.token_tree() {
                    // When matching within a macro expansion, we only want to allow matches of
                    // nodes that originated entirely from within the token tree of the macro call.
                    // i.e. we don't want to match something that came from the macro itself.
                    self.slow_scan_node(
                        &expanded,
                        rule,
                        &Some(self.sema.original_range(tt.syntax())),
                        matches_out,
                    );
                }
            }
        }
        for child in code.children() {
            self.slow_scan_node(&child, rule, restrict_range, matches_out);
        }
    }
}
