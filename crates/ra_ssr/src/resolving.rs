//! This module is responsible for resolving paths within rules.

use crate::errors::error;
use crate::{parsing, SsrError};
use parsing::Placeholder;
use ra_db::FilePosition;
use ra_syntax::{ast, SmolStr, SyntaxKind, SyntaxNode, SyntaxToken};
use rustc_hash::{FxHashMap, FxHashSet};
use test_utils::mark;

pub(crate) struct ResolutionScope<'db> {
    scope: hir::SemanticsScope<'db>,
    hygiene: hir::Hygiene,
}

pub(crate) struct ResolvedRule {
    pub(crate) pattern: ResolvedPattern,
    pub(crate) template: Option<ResolvedPattern>,
    pub(crate) index: usize,
}

pub(crate) struct ResolvedPattern {
    pub(crate) placeholders_by_stand_in: FxHashMap<SmolStr, parsing::Placeholder>,
    pub(crate) node: SyntaxNode,
    // Paths in `node` that we've resolved.
    pub(crate) resolved_paths: FxHashMap<SyntaxNode, ResolvedPath>,
    pub(crate) ufcs_function_calls: FxHashMap<SyntaxNode, hir::Function>,
}

pub(crate) struct ResolvedPath {
    pub(crate) resolution: hir::PathResolution,
    /// The depth of the ast::Path that was resolved within the pattern.
    pub(crate) depth: u32,
}

impl ResolvedRule {
    pub(crate) fn new(
        rule: parsing::ParsedRule,
        resolution_scope: &ResolutionScope,
        index: usize,
    ) -> Result<ResolvedRule, SsrError> {
        let resolver =
            Resolver { resolution_scope, placeholders_by_stand_in: rule.placeholders_by_stand_in };
        let resolved_template = if let Some(template) = rule.template {
            Some(resolver.resolve_pattern_tree(template)?)
        } else {
            None
        };
        Ok(ResolvedRule {
            pattern: resolver.resolve_pattern_tree(rule.pattern)?,
            template: resolved_template,
            index,
        })
    }

    pub(crate) fn get_placeholder(&self, token: &SyntaxToken) -> Option<&Placeholder> {
        if token.kind() != SyntaxKind::IDENT {
            return None;
        }
        self.pattern.placeholders_by_stand_in.get(token.text())
    }
}

struct Resolver<'a, 'db> {
    resolution_scope: &'a ResolutionScope<'db>,
    placeholders_by_stand_in: FxHashMap<SmolStr, parsing::Placeholder>,
}

impl Resolver<'_, '_> {
    fn resolve_pattern_tree(&self, pattern: SyntaxNode) -> Result<ResolvedPattern, SsrError> {
        let mut resolved_paths = FxHashMap::default();
        self.resolve(pattern.clone(), 0, &mut resolved_paths)?;
        let ufcs_function_calls = resolved_paths
            .iter()
            .filter_map(|(path_node, resolved)| {
                if let Some(grandparent) = path_node.parent().and_then(|parent| parent.parent()) {
                    if grandparent.kind() == SyntaxKind::CALL_EXPR {
                        if let hir::PathResolution::AssocItem(hir::AssocItem::Function(function)) =
                            &resolved.resolution
                        {
                            return Some((grandparent, *function));
                        }
                    }
                }
                None
            })
            .collect();
        Ok(ResolvedPattern {
            node: pattern,
            resolved_paths,
            placeholders_by_stand_in: self.placeholders_by_stand_in.clone(),
            ufcs_function_calls,
        })
    }

    fn resolve(
        &self,
        node: SyntaxNode,
        depth: u32,
        resolved_paths: &mut FxHashMap<SyntaxNode, ResolvedPath>,
    ) -> Result<(), SsrError> {
        use ra_syntax::ast::AstNode;
        if let Some(path) = ast::Path::cast(node.clone()) {
            // Check if this is an appropriate place in the path to resolve. If the path is
            // something like `a::B::<i32>::c` then we want to resolve `a::B`. If the path contains
            // a placeholder. e.g. `a::$b::c` then we want to resolve `a`.
            if !path_contains_type_arguments(path.qualifier())
                && !self.path_contains_placeholder(&path)
            {
                let resolution = self
                    .resolution_scope
                    .resolve_path(&path)
                    .ok_or_else(|| error!("Failed to resolve path `{}`", node.text()))?;
                resolved_paths.insert(node, ResolvedPath { resolution, depth });
                return Ok(());
            }
        }
        for node in node.children() {
            self.resolve(node, depth + 1, resolved_paths)?;
        }
        Ok(())
    }

    /// Returns whether `path` contains a placeholder, but ignores any placeholders within type
    /// arguments.
    fn path_contains_placeholder(&self, path: &ast::Path) -> bool {
        if let Some(segment) = path.segment() {
            if let Some(name_ref) = segment.name_ref() {
                if self.placeholders_by_stand_in.contains_key(name_ref.text()) {
                    return true;
                }
            }
        }
        if let Some(qualifier) = path.qualifier() {
            return self.path_contains_placeholder(&qualifier);
        }
        false
    }
}

impl<'db> ResolutionScope<'db> {
    pub(crate) fn new(
        sema: &hir::Semantics<'db, ra_ide_db::RootDatabase>,
        lookup_context: FilePosition,
    ) -> ResolutionScope<'db> {
        use ra_syntax::ast::AstNode;
        let file = sema.parse(lookup_context.file_id);
        // Find a node at the requested position, falling back to the whole file.
        let node = file
            .syntax()
            .token_at_offset(lookup_context.offset)
            .left_biased()
            .map(|token| token.parent())
            .unwrap_or_else(|| file.syntax().clone());
        let node = pick_node_for_resolution(node);
        let scope = sema.scope(&node);
        ResolutionScope {
            scope,
            hygiene: hir::Hygiene::new(sema.db, lookup_context.file_id.into()),
        }
    }

    fn resolve_path(&self, path: &ast::Path) -> Option<hir::PathResolution> {
        let hir_path = hir::Path::from_src(path.clone(), &self.hygiene)?;
        // First try resolving the whole path. This will work for things like
        // `std::collections::HashMap`, but will fail for things like
        // `std::collections::HashMap::new`.
        if let Some(resolution) = self.scope.resolve_hir_path(&hir_path) {
            return Some(resolution);
        }
        // Resolution failed, try resolving the qualifier (e.g. `std::collections::HashMap` and if
        // that succeeds, then iterate through the candidates on the resolved type with the provided
        // name.
        let resolved_qualifier = self.scope.resolve_hir_path_qualifier(&hir_path.qualifier()?)?;
        if let hir::PathResolution::Def(hir::ModuleDef::Adt(adt)) = resolved_qualifier {
            adt.ty(self.scope.db).iterate_path_candidates(
                self.scope.db,
                self.scope.module()?.krate(),
                &FxHashSet::default(),
                Some(hir_path.segments().last()?.name),
                |_ty, assoc_item| Some(hir::PathResolution::AssocItem(assoc_item)),
            )
        } else {
            None
        }
    }
}

/// Returns a suitable node for resolving paths in the current scope. If we create a scope based on
/// a statement node, then we can't resolve local variables that were defined in the current scope
/// (only in parent scopes). So we find another node, ideally a child of the statement where local
/// variable resolution is permitted.
fn pick_node_for_resolution(node: SyntaxNode) -> SyntaxNode {
    match node.kind() {
        SyntaxKind::EXPR_STMT => {
            if let Some(n) = node.first_child() {
                mark::hit!(cursor_after_semicolon);
                return n;
            }
        }
        SyntaxKind::LET_STMT | SyntaxKind::BIND_PAT => {
            if let Some(next) = node.next_sibling() {
                return pick_node_for_resolution(next);
            }
        }
        SyntaxKind::NAME => {
            if let Some(parent) = node.parent() {
                return pick_node_for_resolution(parent);
            }
        }
        _ => {}
    }
    node
}

/// Returns whether `path` or any of its qualifiers contains type arguments.
fn path_contains_type_arguments(path: Option<ast::Path>) -> bool {
    if let Some(path) = path {
        if let Some(segment) = path.segment() {
            if segment.type_arg_list().is_some() {
                mark::hit!(type_arguments_within_path);
                return true;
            }
        }
        return path_contains_type_arguments(path.qualifier());
    }
    false
}
