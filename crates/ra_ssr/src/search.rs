//! Searching for matches.

use crate::{
    matching,
    resolving::{ResolvedPath, ResolvedPattern, ResolvedRule},
    Match, MatchFinder,
};
use ra_db::FileRange;
use ra_ide_db::{
    defs::Definition,
    search::{Reference, SearchScope},
};
use ra_syntax::{ast, AstNode, SyntaxNode};

/// A cache for the results of find_usages. This is for when we have multiple patterns that have the
/// same path. e.g. if the pattern was `foo::Bar` that can parse as a path, an expression, a type
/// and as a pattern. In each, the usages of `foo::Bar` are the same and we'd like to avoid finding
/// them more than once.
#[derive(Default)]
pub(crate) struct UsageCache {
    usages: Vec<(Definition, Vec<Reference>)>,
}

impl<'db> MatchFinder<'db> {
    /// Adds all matches for `rule` to `matches_out`. Matches may overlap in ways that make
    /// replacement impossible, so further processing is required in order to properly nest matches
    /// and remove overlapping matches. This is done in the `nesting` module.
    pub(crate) fn find_matches_for_rule(
        &self,
        rule: &ResolvedRule,
        usage_cache: &mut UsageCache,
        matches_out: &mut Vec<Match>,
    ) {
        if pick_path_for_usages(&rule.pattern).is_none() {
            self.slow_scan(rule, matches_out);
            return;
        }
        self.find_matches_for_pattern_tree(rule, &rule.pattern, usage_cache, matches_out);
    }

    fn find_matches_for_pattern_tree(
        &self,
        rule: &ResolvedRule,
        pattern: &ResolvedPattern,
        usage_cache: &mut UsageCache,
        matches_out: &mut Vec<Match>,
    ) {
        if let Some(first_path) = pick_path_for_usages(pattern) {
            let definition: Definition = first_path.resolution.clone().into();
            for reference in self.find_usages(usage_cache, definition) {
                let file = self.sema.parse(reference.file_range.file_id);
                if let Some(path) = self.sema.find_node_at_offset_with_descend::<ast::Path>(
                    file.syntax(),
                    reference.file_range.range.start(),
                ) {
                    if let Some(node_to_match) = self
                        .sema
                        .ancestors_with_macros(path.syntax().clone())
                        .skip(first_path.depth as usize)
                        .next()
                    {
                        if let Ok(m) =
                            matching::get_match(false, rule, &node_to_match, &None, &self.sema)
                        {
                            matches_out.push(m);
                        }
                    }
                }
            }
        }
    }

    fn find_usages<'a>(
        &self,
        usage_cache: &'a mut UsageCache,
        definition: Definition,
    ) -> &'a [Reference] {
        // Logically if a lookup succeeds we should just return it. Unfortunately returning it would
        // extend the lifetime of the borrow, then we wouldn't be able to do the insertion on a
        // cache miss. This is a limitation of NLL and is fixed with Polonius. For now we do two
        // lookups in the case of a cache hit.
        if usage_cache.find(&definition).is_none() {
            let usages = definition.find_usages(&self.sema, Some(self.search_scope()));
            usage_cache.usages.push((definition, usages));
            return &usage_cache.usages.last().unwrap().1;
        }
        usage_cache.find(&definition).unwrap()
    }

    /// Returns the scope within which we want to search. We don't want un unrestricted search
    /// scope, since we don't want to find references in external dependencies.
    fn search_scope(&self) -> SearchScope {
        // FIXME: We should ideally have a test that checks that we edit local roots and not library
        // roots. This probably would require some changes to fixtures, since currently everything
        // seems to get put into a single source root.
        use ra_db::SourceDatabaseExt;
        use ra_ide_db::symbol_index::SymbolsDatabase;
        let mut files = Vec::new();
        for &root in self.sema.db.local_roots().iter() {
            let sr = self.sema.db.source_root(root);
            files.extend(sr.iter());
        }
        SearchScope::files(&files)
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

impl UsageCache {
    fn find(&mut self, definition: &Definition) -> Option<&[Reference]> {
        // We expect a very small number of cache entries (generally 1), so a linear scan should be
        // fast enough and avoids the need to implement Hash for Definition.
        for (d, refs) in &self.usages {
            if d == definition {
                return Some(refs);
            }
        }
        None
    }
}

/// Returns a path that's suitable for path resolution. We exclude builtin types, since they aren't
/// something that we can find references to. We then somewhat arbitrarily pick the path that is the
/// longest as this is hopefully more likely to be less common, making it faster to find.
fn pick_path_for_usages(pattern: &ResolvedPattern) -> Option<&ResolvedPath> {
    // FIXME: Take the scope of the resolved path into account. e.g. if there are any paths that are
    // private to the current module, then we definitely would want to pick them over say a path
    // from std. Possibly we should go further than this and intersect the search scopes for all
    // resolved paths then search only in that scope.
    pattern
        .resolved_paths
        .iter()
        .filter(|(_, p)| {
            !matches!(p.resolution, hir::PathResolution::Def(hir::ModuleDef::BuiltinType(_)))
        })
        .map(|(node, resolved)| (node.text().len(), resolved))
        .max_by(|(a, _), (b, _)| a.cmp(b))
        .map(|(_, resolved)| resolved)
}
