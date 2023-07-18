//! Searching for matches.

use crate::{
    matching,
    resolving::{ResolvedPath, ResolvedPattern, ResolvedRule},
    Match, MatchFinder,
};
use ide_db::{
    base_db::{FileId, FileRange},
    defs::Definition,
    search::{SearchScope, UsageSearchResult},
    FxHashSet,
};
use syntax::{ast, AstNode, SyntaxKind, SyntaxNode};

/// A cache for the results of find_usages. This is for when we have multiple patterns that have the
/// same path. e.g. if the pattern was `foo::Bar` that can parse as a path, an expression, a type
/// and as a pattern. In each, the usages of `foo::Bar` are the same and we'd like to avoid finding
/// them more than once.
#[derive(Default)]
pub(crate) struct UsageCache {
    usages: Vec<(Definition, UsageSearchResult)>,
}

impl MatchFinder<'_> {
    /// Adds all matches for `rule` to `matches_out`. Matches may overlap in ways that make
    /// replacement impossible, so further processing is required in order to properly nest matches
    /// and remove overlapping matches. This is done in the `nesting` module.
    pub(crate) fn find_matches_for_rule(
        &self,
        rule: &ResolvedRule,
        usage_cache: &mut UsageCache,
        matches_out: &mut Vec<Match>,
    ) {
        if rule.pattern.contains_self {
            // If the pattern contains `self` we restrict the scope of the search to just the
            // current method. No other method can reference the same `self`. This makes the
            // behavior of `self` consistent with other variables.
            if let Some(current_function) = self.resolution_scope.current_function() {
                self.slow_scan_node(&current_function, rule, &None, matches_out);
            }
            return;
        }
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
        if let Some(resolved_path) = pick_path_for_usages(pattern) {
            let definition: Definition = resolved_path.resolution.clone().into();
            for file_range in self.find_usages(usage_cache, definition).file_ranges() {
                for node_to_match in self.find_nodes_to_match(resolved_path, file_range) {
                    if !is_search_permitted_ancestors(&node_to_match) {
                        cov_mark::hit!(use_declaration_with_braces);
                        continue;
                    }
                    self.try_add_match(rule, &node_to_match, &None, matches_out);
                }
            }
        }
    }

    fn find_nodes_to_match(
        &self,
        resolved_path: &ResolvedPath,
        file_range: FileRange,
    ) -> Vec<SyntaxNode> {
        let file = self.sema.parse(file_range.file_id);
        let depth = resolved_path.depth as usize;
        let offset = file_range.range.start();

        let mut paths = self
            .sema
            .find_nodes_at_offset_with_descend::<ast::Path>(file.syntax(), offset)
            .peekable();

        if paths.peek().is_some() {
            paths
                .filter_map(|path| {
                    self.sema.ancestors_with_macros(path.syntax().clone()).nth(depth)
                })
                .collect::<Vec<_>>()
        } else {
            self.sema
                .find_nodes_at_offset_with_descend::<ast::MethodCallExpr>(file.syntax(), offset)
                .filter_map(|path| {
                    // If the pattern contained a path and we found a reference to that path that wasn't
                    // itself a path, but was a method call, then we need to adjust how far up to try
                    // matching by how deep the path was within a CallExpr. The structure would have been
                    // CallExpr, PathExpr, Path - i.e. a depth offset of 2. We don't need to check if the
                    // path was part of a CallExpr because if it wasn't then all that will happen is we'll
                    // fail to match, which is the desired behavior.
                    const PATH_DEPTH_IN_CALL_EXPR: usize = 2;
                    if depth < PATH_DEPTH_IN_CALL_EXPR {
                        return None;
                    }
                    self.sema
                        .ancestors_with_macros(path.syntax().clone())
                        .nth(depth - PATH_DEPTH_IN_CALL_EXPR)
                })
                .collect::<Vec<_>>()
        }
    }

    fn find_usages<'a>(
        &self,
        usage_cache: &'a mut UsageCache,
        definition: Definition,
    ) -> &'a UsageSearchResult {
        // Logically if a lookup succeeds we should just return it. Unfortunately returning it would
        // extend the lifetime of the borrow, then we wouldn't be able to do the insertion on a
        // cache miss. This is a limitation of NLL and is fixed with Polonius. For now we do two
        // lookups in the case of a cache hit.
        if usage_cache.find(&definition).is_none() {
            let usages = definition.usages(&self.sema).in_scope(self.search_scope()).all();
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
        let mut files = Vec::new();
        self.search_files_do(|file_id| {
            files.push(file_id);
        });
        SearchScope::files(&files)
    }

    fn slow_scan(&self, rule: &ResolvedRule, matches_out: &mut Vec<Match>) {
        self.search_files_do(|file_id| {
            let file = self.sema.parse(file_id);
            let code = file.syntax();
            self.slow_scan_node(code, rule, &None, matches_out);
        })
    }

    fn search_files_do(&self, mut callback: impl FnMut(FileId)) {
        if self.restrict_ranges.is_empty() {
            // Unrestricted search.
            use ide_db::base_db::SourceDatabaseExt;
            use ide_db::symbol_index::SymbolsDatabase;
            for &root in self.sema.db.local_roots().iter() {
                let sr = self.sema.db.source_root(root);
                for file_id in sr.iter() {
                    callback(file_id);
                }
            }
        } else {
            // Search is restricted, deduplicate file IDs (generally only one).
            let mut files = FxHashSet::default();
            for range in &self.restrict_ranges {
                if files.insert(range.file_id) {
                    callback(range.file_id);
                }
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
        if !is_search_permitted(code) {
            return;
        }
        self.try_add_match(rule, code, restrict_range, matches_out);
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

    fn try_add_match(
        &self,
        rule: &ResolvedRule,
        code: &SyntaxNode,
        restrict_range: &Option<FileRange>,
        matches_out: &mut Vec<Match>,
    ) {
        if !self.within_range_restrictions(code) {
            cov_mark::hit!(replace_nonpath_within_selection);
            return;
        }
        if let Ok(m) = matching::get_match(false, rule, code, restrict_range, &self.sema) {
            matches_out.push(m);
        }
    }

    /// Returns whether `code` is within one of our range restrictions if we have any. No range
    /// restrictions is considered unrestricted and always returns true.
    fn within_range_restrictions(&self, code: &SyntaxNode) -> bool {
        if self.restrict_ranges.is_empty() {
            // There is no range restriction.
            return true;
        }
        let node_range = self.sema.original_range(code);
        for range in &self.restrict_ranges {
            if range.file_id == node_range.file_id && range.range.contains_range(node_range.range) {
                return true;
            }
        }
        false
    }
}

/// Returns whether we support matching within `node` and all of its ancestors.
fn is_search_permitted_ancestors(node: &SyntaxNode) -> bool {
    if let Some(parent) = node.parent() {
        if !is_search_permitted_ancestors(&parent) {
            return false;
        }
    }
    is_search_permitted(node)
}

/// Returns whether we support matching within this kind of node.
fn is_search_permitted(node: &SyntaxNode) -> bool {
    // FIXME: Properly handle use declarations. At the moment, if our search pattern is `foo::bar`
    // and the code is `use foo::{baz, bar}`, we'll match `bar`, since it resolves to `foo::bar`.
    // However we'll then replace just the part we matched `bar`. We probably need to instead remove
    // `bar` and insert a new use declaration.
    node.kind() != SyntaxKind::USE
}

impl UsageCache {
    fn find(&mut self, definition: &Definition) -> Option<&UsageSearchResult> {
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
