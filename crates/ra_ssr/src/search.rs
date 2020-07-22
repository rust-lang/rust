//! Searching for matches.

use crate::{matching, Match, MatchFinder};
use ra_db::FileRange;
use ra_syntax::{ast, AstNode, SyntaxNode};

impl<'db> MatchFinder<'db> {
    pub(crate) fn find_all_matches(&self, matches_out: &mut Vec<Match>) {
        // FIXME: Use resolved paths in the pattern to find places to search instead of always
        // scanning every node.
        self.slow_scan(matches_out);
    }

    fn slow_scan(&self, matches_out: &mut Vec<Match>) {
        use ra_db::SourceDatabaseExt;
        use ra_ide_db::symbol_index::SymbolsDatabase;
        for &root in self.sema.db.local_roots().iter() {
            let sr = self.sema.db.source_root(root);
            for file_id in sr.iter() {
                let file = self.sema.parse(file_id);
                let code = file.syntax();
                self.slow_scan_node(code, &None, matches_out);
            }
        }
    }

    fn slow_scan_node(
        &self,
        code: &SyntaxNode,
        restrict_range: &Option<FileRange>,
        matches_out: &mut Vec<Match>,
    ) {
        for rule in &self.rules {
            if let Ok(mut m) = matching::get_match(false, rule, &code, restrict_range, &self.sema) {
                // Continue searching in each of our placeholders.
                for placeholder_value in m.placeholder_values.values_mut() {
                    if let Some(placeholder_node) = &placeholder_value.node {
                        // Don't search our placeholder if it's the entire matched node, otherwise we'd
                        // find the same match over and over until we got a stack overflow.
                        if placeholder_node != code {
                            self.slow_scan_node(
                                placeholder_node,
                                restrict_range,
                                &mut placeholder_value.inner_matches.matches,
                            );
                        }
                    }
                }
                matches_out.push(m);
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
                    self.slow_scan_node(
                        &expanded,
                        &Some(self.sema.original_range(tt.syntax())),
                        matches_out,
                    );
                }
            }
        }
        for child in code.children() {
            self.slow_scan_node(&child, restrict_range, matches_out);
        }
    }
}
