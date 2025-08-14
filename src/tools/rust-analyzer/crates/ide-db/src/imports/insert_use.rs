//! Handle syntactic aspects of inserting a new `use` item.
#[cfg(test)]
mod tests;

use std::cmp::Ordering;

use hir::Semantics;
use syntax::{
    Direction, NodeOrToken, SyntaxKind, SyntaxNode, algo,
    ast::{
        self, AstNode, HasAttrs, HasModuleItem, HasVisibility, PathSegmentKind,
        edit_in_place::Removable, make,
    },
    ted,
};

use crate::{
    RootDatabase,
    imports::merge_imports::{
        MergeBehavior, NormalizationStyle, common_prefix, eq_attrs, eq_visibility,
        try_merge_imports, use_tree_cmp,
    },
};

pub use hir::PrefixKind;

/// How imports should be grouped into use statements.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ImportGranularity {
    /// Do not change the granularity of any imports and preserve the original structure written
    /// by the developer.
    Preserve,
    /// Merge imports from the same crate into a single use statement.
    Crate,
    /// Merge imports from the same module into a single use statement.
    Module,
    /// Flatten imports so that each has its own use statement.
    Item,
    /// Merge all imports into a single use statement as long as they have the same visibility
    /// and attributes.
    One,
}

impl From<ImportGranularity> for NormalizationStyle {
    fn from(granularity: ImportGranularity) -> Self {
        match granularity {
            ImportGranularity::One => NormalizationStyle::One,
            _ => NormalizationStyle::Default,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InsertUseConfig {
    pub granularity: ImportGranularity,
    pub enforce_granularity: bool,
    pub prefix_kind: PrefixKind,
    pub group: bool,
    pub skip_glob_imports: bool,
}

#[derive(Debug, Clone)]
pub struct ImportScope {
    pub kind: ImportScopeKind,
    pub required_cfgs: Vec<ast::Attr>,
}

#[derive(Debug, Clone)]
pub enum ImportScopeKind {
    File(ast::SourceFile),
    Module(ast::ItemList),
    Block(ast::StmtList),
}

impl ImportScope {
    /// Determines the containing syntax node in which to insert a `use` statement affecting `position`.
    /// Returns the original source node inside attributes.
    pub fn find_insert_use_container(
        position: &SyntaxNode,
        sema: &Semantics<'_, RootDatabase>,
    ) -> Option<Self> {
        // The closest block expression ancestor
        let mut block = None;
        let mut required_cfgs = Vec::new();
        // Walk up the ancestor tree searching for a suitable node to do insertions on
        // with special handling on cfg-gated items, in which case we want to insert imports locally
        // or FIXME: annotate inserted imports with the same cfg
        for syntax in sema.ancestors_with_macros(position.clone()) {
            if let Some(file) = ast::SourceFile::cast(syntax.clone()) {
                return Some(ImportScope { kind: ImportScopeKind::File(file), required_cfgs });
            } else if let Some(module) = ast::Module::cast(syntax.clone()) {
                // early return is important here, if we can't find the original module
                // in the input there is no way for us to insert an import anywhere.
                return sema
                    .original_ast_node(module)?
                    .item_list()
                    .map(ImportScopeKind::Module)
                    .map(|kind| ImportScope { kind, required_cfgs });
            } else if let Some(has_attrs) = ast::AnyHasAttrs::cast(syntax) {
                if block.is_none()
                    && let Some(b) = ast::BlockExpr::cast(has_attrs.syntax().clone())
                    && let Some(b) = sema.original_ast_node(b)
                {
                    block = b.stmt_list();
                }
                if has_attrs
                    .attrs()
                    .any(|attr| attr.as_simple_call().is_some_and(|(ident, _)| ident == "cfg"))
                {
                    if let Some(b) = block {
                        return Some(ImportScope {
                            kind: ImportScopeKind::Block(b),
                            required_cfgs,
                        });
                    }
                    required_cfgs.extend(has_attrs.attrs().filter(|attr| {
                        attr.as_simple_call().is_some_and(|(ident, _)| ident == "cfg")
                    }));
                }
            }
        }
        None
    }

    pub fn as_syntax_node(&self) -> &SyntaxNode {
        match &self.kind {
            ImportScopeKind::File(file) => file.syntax(),
            ImportScopeKind::Module(item_list) => item_list.syntax(),
            ImportScopeKind::Block(block) => block.syntax(),
        }
    }

    pub fn clone_for_update(&self) -> Self {
        Self {
            kind: match &self.kind {
                ImportScopeKind::File(file) => ImportScopeKind::File(file.clone_for_update()),
                ImportScopeKind::Module(item_list) => {
                    ImportScopeKind::Module(item_list.clone_for_update())
                }
                ImportScopeKind::Block(block) => ImportScopeKind::Block(block.clone_for_update()),
            },
            required_cfgs: self.required_cfgs.iter().map(|attr| attr.clone_for_update()).collect(),
        }
    }
}

/// Insert an import path into the given file/node. A `merge` value of none indicates that no import merging is allowed to occur.
pub fn insert_use(scope: &ImportScope, path: ast::Path, cfg: &InsertUseConfig) {
    insert_use_with_alias_option(scope, path, cfg, None);
}

pub fn insert_use_as_alias(scope: &ImportScope, path: ast::Path, cfg: &InsertUseConfig) {
    let text: &str = "use foo as _";
    let parse = syntax::SourceFile::parse(text, span::Edition::CURRENT_FIXME);
    let node = parse
        .tree()
        .syntax()
        .descendants()
        .find_map(ast::UseTree::cast)
        .expect("Failed to make ast node `Rename`");
    let alias = node.rename();

    insert_use_with_alias_option(scope, path, cfg, alias);
}

fn insert_use_with_alias_option(
    scope: &ImportScope,
    path: ast::Path,
    cfg: &InsertUseConfig,
    alias: Option<ast::Rename>,
) {
    let _p = tracing::info_span!("insert_use_with_alias_option").entered();
    let mut mb = match cfg.granularity {
        ImportGranularity::Crate => Some(MergeBehavior::Crate),
        ImportGranularity::Module => Some(MergeBehavior::Module),
        ImportGranularity::One => Some(MergeBehavior::One),
        ImportGranularity::Item | ImportGranularity::Preserve => None,
    };
    if !cfg.enforce_granularity {
        let file_granularity = guess_granularity_from_scope(scope);
        mb = match file_granularity {
            ImportGranularityGuess::Unknown => mb,
            ImportGranularityGuess::Item => None,
            ImportGranularityGuess::Module => Some(MergeBehavior::Module),
            ImportGranularityGuess::ModuleOrItem => mb.and(Some(MergeBehavior::Module)),
            ImportGranularityGuess::Crate => Some(MergeBehavior::Crate),
            ImportGranularityGuess::CrateOrModule => mb.or(Some(MergeBehavior::Crate)),
            ImportGranularityGuess::One => Some(MergeBehavior::One),
        };
    }

    let mut use_tree = make::use_tree(path, None, alias, false);
    if mb == Some(MergeBehavior::One) && use_tree.path().is_some() {
        use_tree = use_tree.clone_for_update();
        use_tree.wrap_in_tree_list();
    }
    let use_item = make::use_(None, use_tree).clone_for_update();
    for attr in
        scope.required_cfgs.iter().map(|attr| attr.syntax().clone_subtree().clone_for_update())
    {
        ted::insert(ted::Position::first_child_of(use_item.syntax()), attr);
    }

    // merge into existing imports if possible
    if let Some(mb) = mb {
        let filter = |it: &_| !(cfg.skip_glob_imports && ast::Use::is_simple_glob(it));
        for existing_use in
            scope.as_syntax_node().children().filter_map(ast::Use::cast).filter(filter)
        {
            if let Some(merged) = try_merge_imports(&existing_use, &use_item, mb) {
                ted::replace(existing_use.syntax(), merged.syntax());
                return;
            }
        }
    }
    // either we weren't allowed to merge or there is no import that fits the merge conditions
    // so look for the place we have to insert to
    insert_use_(scope, use_item, cfg.group);
}

pub fn ast_to_remove_for_path_in_use_stmt(path: &ast::Path) -> Option<Box<dyn Removable>> {
    // FIXME: improve this
    if path.parent_path().is_some() {
        return None;
    }
    let use_tree = path.syntax().parent().and_then(ast::UseTree::cast)?;
    if use_tree.use_tree_list().is_some() || use_tree.star_token().is_some() {
        return None;
    }
    if let Some(use_) = use_tree.syntax().parent().and_then(ast::Use::cast) {
        return Some(Box::new(use_));
    }
    Some(Box::new(use_tree))
}

pub fn remove_path_if_in_use_stmt(path: &ast::Path) {
    if let Some(node) = ast_to_remove_for_path_in_use_stmt(path) {
        node.remove();
    }
}

#[derive(Eq, PartialEq, PartialOrd, Ord)]
enum ImportGroup {
    // the order here defines the order of new group inserts
    Std,
    ExternCrate,
    ThisCrate,
    ThisModule,
    SuperModule,
    One,
}

impl ImportGroup {
    fn new(use_tree: &ast::UseTree) -> ImportGroup {
        if use_tree.path().is_none() && use_tree.use_tree_list().is_some() {
            return ImportGroup::One;
        }

        let Some(first_segment) = use_tree.path().as_ref().and_then(ast::Path::first_segment)
        else {
            return ImportGroup::ExternCrate;
        };

        let kind = first_segment.kind().unwrap_or(PathSegmentKind::SelfKw);
        match kind {
            PathSegmentKind::SelfKw => ImportGroup::ThisModule,
            PathSegmentKind::SuperKw => ImportGroup::SuperModule,
            PathSegmentKind::CrateKw => ImportGroup::ThisCrate,
            PathSegmentKind::Name(name) => match name.text().as_str() {
                "std" => ImportGroup::Std,
                "core" => ImportGroup::Std,
                _ => ImportGroup::ExternCrate,
            },
            // these aren't valid use paths, so fall back to something random
            PathSegmentKind::SelfTypeKw => ImportGroup::ExternCrate,
            PathSegmentKind::Type { .. } => ImportGroup::ExternCrate,
        }
    }
}

#[derive(PartialEq, PartialOrd, Debug, Clone, Copy)]
enum ImportGranularityGuess {
    Unknown,
    Item,
    Module,
    ModuleOrItem,
    Crate,
    CrateOrModule,
    One,
}

fn guess_granularity_from_scope(scope: &ImportScope) -> ImportGranularityGuess {
    // The idea is simple, just check each import as well as the import and its precedent together for
    // whether they fulfill a granularity criteria.
    let use_stmt = |item| match item {
        ast::Item::Use(use_) => {
            let use_tree = use_.use_tree()?;
            Some((use_tree, use_.visibility(), use_.attrs()))
        }
        _ => None,
    };
    let mut use_stmts = match &scope.kind {
        ImportScopeKind::File(f) => f.items(),
        ImportScopeKind::Module(m) => m.items(),
        ImportScopeKind::Block(b) => b.items(),
    }
    .filter_map(use_stmt);
    let mut res = ImportGranularityGuess::Unknown;
    let Some((mut prev, mut prev_vis, mut prev_attrs)) = use_stmts.next() else { return res };

    let is_tree_one_style =
        |use_tree: &ast::UseTree| use_tree.path().is_none() && use_tree.use_tree_list().is_some();
    let mut seen_one_style_groups = Vec::new();

    loop {
        if is_tree_one_style(&prev) {
            if res != ImportGranularityGuess::One {
                if res != ImportGranularityGuess::Unknown {
                    // This scope has a mix of one-style and other style imports.
                    break ImportGranularityGuess::Unknown;
                }

                res = ImportGranularityGuess::One;
                seen_one_style_groups.push((prev_vis.clone(), prev_attrs.clone()));
            }
        } else if let Some(use_tree_list) = prev.use_tree_list() {
            if use_tree_list.use_trees().any(|tree| tree.use_tree_list().is_some()) {
                // Nested tree lists can only occur in crate style, or with no proper style being enforced in the file.
                break ImportGranularityGuess::Crate;
            } else {
                // Could still be crate-style so continue looking.
                res = ImportGranularityGuess::CrateOrModule;
            }
        }

        let Some((curr, curr_vis, curr_attrs)) = use_stmts.next() else { break res };
        if is_tree_one_style(&curr) {
            if res != ImportGranularityGuess::One
                || seen_one_style_groups.iter().any(|(prev_vis, prev_attrs)| {
                    eq_visibility(prev_vis.clone(), curr_vis.clone())
                        && eq_attrs(prev_attrs.clone(), curr_attrs.clone())
                })
            {
                // This scope has either a mix of one-style and other style imports or
                // multiple one-style imports with the same visibility and attributes.
                break ImportGranularityGuess::Unknown;
            }
            seen_one_style_groups.push((curr_vis.clone(), curr_attrs.clone()));
        } else if eq_visibility(prev_vis, curr_vis.clone())
            && eq_attrs(prev_attrs, curr_attrs.clone())
            && let Some((prev_path, curr_path)) = prev.path().zip(curr.path())
            && let Some((prev_prefix, _)) = common_prefix(&prev_path, &curr_path)
        {
            if prev.use_tree_list().is_none() && curr.use_tree_list().is_none() {
                let prefix_c = prev_prefix.qualifiers().count();
                let curr_c = curr_path.qualifiers().count() - prefix_c;
                let prev_c = prev_path.qualifiers().count() - prefix_c;
                if curr_c == 1 && prev_c == 1 {
                    // Same prefix, only differing in the last segment and no use tree lists so this has to be of item style.
                    break ImportGranularityGuess::Item;
                } else {
                    // Same prefix and no use tree list but differs in more than one segment at the end. This might be module style still.
                    res = ImportGranularityGuess::ModuleOrItem;
                }
            } else {
                // Same prefix with item tree lists, has to be module style as it
                // can't be crate style since the trees wouldn't share a prefix then.
                break ImportGranularityGuess::Module;
            }
        }
        prev = curr;
        prev_vis = curr_vis;
        prev_attrs = curr_attrs;
    }
}

fn insert_use_(scope: &ImportScope, use_item: ast::Use, group_imports: bool) {
    let scope_syntax = scope.as_syntax_node();
    let insert_use_tree =
        use_item.use_tree().expect("`use_item` should have a use tree for `insert_path`");
    let group = ImportGroup::new(&insert_use_tree);
    let path_node_iter = scope_syntax
        .children()
        .filter_map(|node| ast::Use::cast(node.clone()).zip(Some(node)))
        .flat_map(|(use_, node)| {
            let tree = use_.use_tree()?;
            Some((tree, node))
        });

    if group_imports {
        // Iterator that discards anything that's not in the required grouping
        // This implementation allows the user to rearrange their import groups as this only takes the first group that fits
        let group_iter = path_node_iter
            .clone()
            .skip_while(|(use_tree, ..)| ImportGroup::new(use_tree) != group)
            .take_while(|(use_tree, ..)| ImportGroup::new(use_tree) == group);

        // track the last element we iterated over, if this is still None after the iteration then that means we never iterated in the first place
        let mut last = None;
        // find the element that would come directly after our new import
        let post_insert: Option<(_, SyntaxNode)> = group_iter
            .inspect(|(.., node)| last = Some(node.clone()))
            .find(|(use_tree, _)| use_tree_cmp(&insert_use_tree, use_tree) != Ordering::Greater);

        if let Some((.., node)) = post_insert {
            cov_mark::hit!(insert_group);
            // insert our import before that element
            return ted::insert(ted::Position::before(node), use_item.syntax());
        }
        if let Some(node) = last {
            cov_mark::hit!(insert_group_last);
            // there is no element after our new import, so append it to the end of the group
            return ted::insert(ted::Position::after(node), use_item.syntax());
        }

        // the group we were looking for actually doesn't exist, so insert

        let mut last = None;
        // find the group that comes after where we want to insert
        let post_group = path_node_iter
            .inspect(|(.., node)| last = Some(node.clone()))
            .find(|(use_tree, ..)| ImportGroup::new(use_tree) > group);
        if let Some((.., node)) = post_group {
            cov_mark::hit!(insert_group_new_group);
            ted::insert(ted::Position::before(&node), use_item.syntax());
            if let Some(node) = algo::non_trivia_sibling(node.into(), Direction::Prev) {
                ted::insert(ted::Position::after(node), make::tokens::single_newline());
            }
            return;
        }
        // there is no such group, so append after the last one
        if let Some(node) = last {
            cov_mark::hit!(insert_group_no_group);
            ted::insert(ted::Position::after(&node), use_item.syntax());
            ted::insert(ted::Position::after(node), make::tokens::single_newline());
            return;
        }
    } else {
        // There exists a group, so append to the end of it
        if let Some((_, node)) = path_node_iter.last() {
            cov_mark::hit!(insert_no_grouping_last);
            ted::insert(ted::Position::after(node), use_item.syntax());
            return;
        }
    }

    let l_curly = match &scope.kind {
        ImportScopeKind::File(_) => None,
        // don't insert the imports before the item list/block expr's opening curly brace
        ImportScopeKind::Module(item_list) => item_list.l_curly_token(),
        // don't insert the imports before the item list's opening curly brace
        ImportScopeKind::Block(block) => block.l_curly_token(),
    };
    // there are no imports in this file at all
    // so put the import after all inner module attributes and possible license header comments
    if let Some(last_inner_element) = scope_syntax
        .children_with_tokens()
        // skip the curly brace
        .skip(l_curly.is_some() as usize)
        .take_while(|child| match child {
            NodeOrToken::Node(node) => is_inner_attribute(node.clone()),
            NodeOrToken::Token(token) => {
                [SyntaxKind::WHITESPACE, SyntaxKind::COMMENT, SyntaxKind::SHEBANG]
                    .contains(&token.kind())
            }
        })
        .filter(|child| child.as_token().is_none_or(|t| t.kind() != SyntaxKind::WHITESPACE))
        .last()
    {
        cov_mark::hit!(insert_empty_inner_attr);
        ted::insert(ted::Position::after(&last_inner_element), use_item.syntax());
        ted::insert(ted::Position::after(last_inner_element), make::tokens::single_newline());
    } else {
        match l_curly {
            Some(b) => {
                cov_mark::hit!(insert_empty_module);
                ted::insert(ted::Position::after(&b), make::tokens::single_newline());
                ted::insert(ted::Position::after(&b), use_item.syntax());
            }
            None => {
                cov_mark::hit!(insert_empty_file);
                ted::insert(
                    ted::Position::first_child_of(scope_syntax),
                    make::tokens::blank_line(),
                );
                ted::insert(ted::Position::first_child_of(scope_syntax), use_item.syntax());
            }
        }
    }
}

fn is_inner_attribute(node: SyntaxNode) -> bool {
    ast::Attr::cast(node).map(|attr| attr.kind()) == Some(ast::AttrKind::Inner)
}
