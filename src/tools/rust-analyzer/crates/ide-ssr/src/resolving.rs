//! This module is responsible for resolving paths within rules.

use hir::AsAssocItem;
use ide_db::FxHashMap;
use parsing::Placeholder;
use syntax::{
    SmolStr, SyntaxKind, SyntaxNode, SyntaxToken,
    ast::{self, HasGenericArgs},
};

use crate::{SsrError, errors::error, parsing};

pub(crate) struct ResolutionScope<'db> {
    scope: hir::SemanticsScope<'db>,
    node: SyntaxNode,
}

pub(crate) struct ResolvedRule<'db> {
    pub(crate) pattern: ResolvedPattern<'db>,
    pub(crate) template: Option<ResolvedPattern<'db>>,
    pub(crate) index: usize,
}

pub(crate) struct ResolvedPattern<'db> {
    pub(crate) placeholders_by_stand_in: FxHashMap<SmolStr, parsing::Placeholder>,
    pub(crate) node: SyntaxNode,
    // Paths in `node` that we've resolved.
    pub(crate) resolved_paths: FxHashMap<SyntaxNode, ResolvedPath>,
    pub(crate) ufcs_function_calls: FxHashMap<SyntaxNode, UfcsCallInfo<'db>>,
    pub(crate) contains_self: bool,
}

pub(crate) struct ResolvedPath {
    pub(crate) resolution: hir::PathResolution,
    /// The depth of the ast::Path that was resolved within the pattern.
    pub(crate) depth: u32,
}

pub(crate) struct UfcsCallInfo<'db> {
    pub(crate) call_expr: ast::CallExpr,
    pub(crate) function: hir::Function,
    pub(crate) qualifier_type: Option<hir::Type<'db>>,
}

impl<'db> ResolvedRule<'db> {
    pub(crate) fn new(
        rule: parsing::ParsedRule,
        resolution_scope: &ResolutionScope<'db>,
        index: usize,
    ) -> Result<ResolvedRule<'db>, SsrError> {
        let resolver =
            Resolver { resolution_scope, placeholders_by_stand_in: rule.placeholders_by_stand_in };
        let resolved_template = match rule.template {
            Some(template) => Some(resolver.resolve_pattern_tree(template)?),
            None => None,
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

impl<'db> Resolver<'_, 'db> {
    fn resolve_pattern_tree(&self, pattern: SyntaxNode) -> Result<ResolvedPattern<'db>, SsrError> {
        use syntax::ast::AstNode;
        use syntax::{SyntaxElement, T};
        let mut resolved_paths = FxHashMap::default();
        self.resolve(pattern.clone(), 0, &mut resolved_paths)?;
        let ufcs_function_calls = resolved_paths
            .iter()
            .filter_map(|(path_node, resolved)| {
                if let Some(grandparent) = path_node.parent().and_then(|parent| parent.parent()) {
                    if let Some(call_expr) = ast::CallExpr::cast(grandparent.clone()) {
                        if let hir::PathResolution::Def(hir::ModuleDef::Function(function)) =
                            resolved.resolution
                        {
                            if function.as_assoc_item(self.resolution_scope.scope.db).is_some() {
                                let qualifier_type =
                                    self.resolution_scope.qualifier_type(path_node);
                                return Some((
                                    grandparent,
                                    UfcsCallInfo { call_expr, function, qualifier_type },
                                ));
                            }
                        }
                    }
                }
                None
            })
            .collect();
        let contains_self =
            pattern.descendants_with_tokens().any(|node_or_token| match node_or_token {
                SyntaxElement::Token(t) => t.kind() == T![self],
                _ => false,
            });
        Ok(ResolvedPattern {
            node: pattern,
            resolved_paths,
            placeholders_by_stand_in: self.placeholders_by_stand_in.clone(),
            ufcs_function_calls,
            contains_self,
        })
    }

    fn resolve(
        &self,
        node: SyntaxNode,
        depth: u32,
        resolved_paths: &mut FxHashMap<SyntaxNode, ResolvedPath>,
    ) -> Result<(), SsrError> {
        use syntax::ast::AstNode;
        if let Some(path) = ast::Path::cast(node.clone()) {
            if is_self(&path) {
                // Self cannot be resolved like other paths.
                return Ok(());
            }
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
                if self.ok_to_use_path_resolution(&resolution) {
                    resolved_paths.insert(node, ResolvedPath { resolution, depth });
                    return Ok(());
                }
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
                if self.placeholders_by_stand_in.contains_key(name_ref.text().as_str()) {
                    return true;
                }
            }
        }
        if let Some(qualifier) = path.qualifier() {
            return self.path_contains_placeholder(&qualifier);
        }
        false
    }

    fn ok_to_use_path_resolution(&self, resolution: &hir::PathResolution) -> bool {
        match resolution {
            hir::PathResolution::Def(hir::ModuleDef::Function(function))
                if function.as_assoc_item(self.resolution_scope.scope.db).is_some() =>
            {
                if function.self_param(self.resolution_scope.scope.db).is_some() {
                    // If we don't use this path resolution, then we won't be able to match method
                    // calls. e.g. `Foo::bar($s)` should match `x.bar()`.
                    true
                } else {
                    cov_mark::hit!(replace_associated_trait_default_function_call);
                    false
                }
            }
            hir::PathResolution::Def(
                def @ (hir::ModuleDef::Const(_) | hir::ModuleDef::TypeAlias(_)),
            ) if def.as_assoc_item(self.resolution_scope.scope.db).is_some() => {
                // Not a function. Could be a constant or an associated type.
                cov_mark::hit!(replace_associated_trait_constant);
                false
            }
            _ => true,
        }
    }
}

impl<'db> ResolutionScope<'db> {
    pub(crate) fn new(
        sema: &hir::Semantics<'db, ide_db::RootDatabase>,
        resolve_context: hir::FilePosition,
    ) -> Option<ResolutionScope<'db>> {
        use syntax::ast::AstNode;
        let file = sema.parse(resolve_context.file_id);
        // Find a node at the requested position, falling back to the whole file.
        let node = file
            .syntax()
            .token_at_offset(resolve_context.offset)
            .left_biased()
            .and_then(|token| token.parent())
            .unwrap_or_else(|| file.syntax().clone());
        let node = pick_node_for_resolution(node);
        let scope = sema.scope(&node)?;
        Some(ResolutionScope { scope, node })
    }

    /// Returns the function in which SSR was invoked, if any.
    pub(crate) fn current_function(&self) -> Option<SyntaxNode> {
        self.node.ancestors().find(|node| node.kind() == SyntaxKind::FN)
    }

    fn resolve_path(&self, path: &ast::Path) -> Option<hir::PathResolution> {
        // First try resolving the whole path. This will work for things like
        // `std::collections::HashMap`, but will fail for things like
        // `std::collections::HashMap::new`.
        if let Some(resolution) = self.scope.speculative_resolve(path) {
            return Some(resolution);
        }
        // Resolution failed, try resolving the qualifier (e.g. `std::collections::HashMap` and if
        // that succeeds, then iterate through the candidates on the resolved type with the provided
        // name.
        let resolved_qualifier = self.scope.speculative_resolve(&path.qualifier()?)?;
        if let hir::PathResolution::Def(hir::ModuleDef::Adt(adt)) = resolved_qualifier {
            let name = path.segment()?.name_ref()?;
            let module = self.scope.module();
            adt.ty(self.scope.db).iterate_path_candidates(
                self.scope.db,
                &self.scope,
                &self.scope.visible_traits().0,
                Some(module),
                None,
                |assoc_item| {
                    let item_name = assoc_item.name(self.scope.db)?;
                    if item_name.as_str() == name.text() {
                        Some(hir::PathResolution::Def(assoc_item.into()))
                    } else {
                        None
                    }
                },
            )
        } else {
            None
        }
    }

    fn qualifier_type(&self, path: &SyntaxNode) -> Option<hir::Type<'db>> {
        use syntax::ast::AstNode;
        if let Some(path) = ast::Path::cast(path.clone()) {
            if let Some(qualifier) = path.qualifier() {
                if let Some(hir::PathResolution::Def(hir::ModuleDef::Adt(adt))) =
                    self.resolve_path(&qualifier)
                {
                    return Some(adt.ty(self.scope.db));
                }
            }
        }
        None
    }
}

fn is_self(path: &ast::Path) -> bool {
    path.segment().map(|segment| segment.self_token().is_some()).unwrap_or(false)
}

/// Returns a suitable node for resolving paths in the current scope. If we create a scope based on
/// a statement node, then we can't resolve local variables that were defined in the current scope
/// (only in parent scopes). So we find another node, ideally a child of the statement where local
/// variable resolution is permitted.
fn pick_node_for_resolution(node: SyntaxNode) -> SyntaxNode {
    match node.kind() {
        SyntaxKind::EXPR_STMT => {
            if let Some(n) = node.first_child() {
                cov_mark::hit!(cursor_after_semicolon);
                return n;
            }
        }
        SyntaxKind::LET_STMT | SyntaxKind::IDENT_PAT => {
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
            if segment.generic_arg_list().is_some() {
                cov_mark::hit!(type_arguments_within_path);
                return true;
            }
        }
        return path_contains_type_arguments(path.qualifier());
    }
    false
}
