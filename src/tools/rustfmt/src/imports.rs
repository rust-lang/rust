use std::borrow::Cow;
use std::cmp::Ordering;
use std::fmt;

use core::hash::{Hash, Hasher};

use itertools::Itertools;

use rustc_ast::ast::{self, UseTreeKind};
use rustc_span::{
    BytePos, DUMMY_SP, Span,
    symbol::{self, sym},
};

use crate::comment::combine_strs_with_missing_comments;
use crate::config::ImportGranularity;
use crate::config::lists::*;
use crate::config::{Edition, IndentStyle, StyleEdition};
use crate::lists::{
    ListFormatting, ListItem, Separator, definitive_tactic, itemize_list, write_list,
};
use crate::rewrite::{Rewrite, RewriteContext, RewriteErrorExt, RewriteResult};
use crate::shape::Shape;
use crate::sort::version_sort;
use crate::source_map::SpanUtils;
use crate::spanned::Spanned;
use crate::utils::{is_same_visibility, mk_sp, rewrite_ident};
use crate::visitor::FmtVisitor;

/// Returns a name imported by a `use` declaration.
/// E.g., returns `Ordering` for `std::cmp::Ordering` and `self` for `std::cmp::self`.
pub(crate) fn path_to_imported_ident(path: &ast::Path) -> symbol::Ident {
    path.segments.last().unwrap().ident
}

impl<'a> FmtVisitor<'a> {
    pub(crate) fn format_import(&mut self, item: &ast::Item, tree: &ast::UseTree) {
        let span = item.span();
        let shape = self.shape();
        let rw = UseTree::from_ast(
            &self.get_context(),
            tree,
            None,
            Some(item.vis.clone()),
            Some(item.span.lo()),
            Some(item.attrs.clone()),
        )
        .rewrite_top_level(&self.get_context(), shape)
        .ok();
        match rw {
            Some(ref s) if s.is_empty() => {
                // Format up to last newline
                let prev_span = mk_sp(self.last_pos, source!(self, span).lo());
                let trimmed_snippet = self.snippet(prev_span).trim_end();
                let span_end = self.last_pos + BytePos(trimmed_snippet.len() as u32);
                self.format_missing(span_end);
                // We have an excessive newline from the removed import.
                if self.buffer.ends_with('\n') {
                    self.buffer.pop();
                    self.line_number -= 1;
                }
                self.last_pos = source!(self, span).hi();
            }
            Some(ref s) => {
                self.format_missing_with_indent(source!(self, span).lo());
                self.push_str(s);
                self.last_pos = source!(self, span).hi();
            }
            None => {
                self.format_missing_with_indent(source!(self, span).lo());
                self.format_missing(source!(self, span).hi());
            }
        }
    }
}

// Ordering of imports

// We order imports by translating to our own representation and then sorting.
// The Rust AST data structures are really bad for this. Rustfmt applies a bunch
// of normalisations to imports and since we want to sort based on the result
// of these (and to maintain idempotence) we must apply the same normalisations
// to the data structures for sorting.
//
// We sort `self` and `super` before other imports, then identifier imports,
// then glob imports, then lists of imports. We do not take aliases into account
// when ordering unless the imports are identical except for the alias (rare in
// practice).

// FIXME(#2531): we should unify the comparison code here with the formatting
// code elsewhere since we are essentially string-ifying twice. Furthermore, by
// parsing to our own format on comparison, we repeat a lot of work when
// sorting.

// FIXME we do a lot of allocation to make our own representation.
#[derive(Clone, Eq, Hash, PartialEq)]
pub(crate) enum UseSegmentKind {
    Ident(String, Option<String>),
    Slf(Option<String>),
    Super(Option<String>),
    Crate(Option<String>),
    Glob,
    List(Vec<UseTree>),
}

#[derive(Clone, Eq, PartialEq)]
pub(crate) struct UseSegment {
    pub(crate) kind: UseSegmentKind,
    pub(crate) style_edition: StyleEdition,
}

#[derive(Clone)]
pub(crate) struct UseTree {
    pub(crate) path: Vec<UseSegment>,
    pub(crate) span: Span,
    // Comment information within nested use tree.
    pub(crate) list_item: Option<ListItem>,
    // Additional fields for top level use items.
    // Should we have another struct for top-level use items rather than reusing this?
    visibility: Option<ast::Visibility>,
    attrs: Option<ast::AttrVec>,
}

impl PartialEq for UseTree {
    fn eq(&self, other: &UseTree) -> bool {
        self.path == other.path
    }
}
impl Eq for UseTree {}

impl Spanned for UseTree {
    fn span(&self) -> Span {
        let lo = if let Some(ref attrs) = self.attrs {
            attrs.iter().next().map_or(self.span.lo(), |a| a.span.lo())
        } else {
            self.span.lo()
        };
        mk_sp(lo, self.span.hi())
    }
}

impl UseSegment {
    // Clone a version of self with any top-level alias removed.
    fn remove_alias(&self) -> UseSegment {
        let kind = match self.kind {
            UseSegmentKind::Ident(ref s, _) => UseSegmentKind::Ident(s.clone(), None),
            UseSegmentKind::Slf(_) => UseSegmentKind::Slf(None),
            UseSegmentKind::Super(_) => UseSegmentKind::Super(None),
            UseSegmentKind::Crate(_) => UseSegmentKind::Crate(None),
            _ => return self.clone(),
        };
        UseSegment {
            kind,
            style_edition: self.style_edition,
        }
    }

    // Check if self == other with their aliases removed.
    fn equal_except_alias(&self, other: &Self) -> bool {
        match (&self.kind, &other.kind) {
            (UseSegmentKind::Ident(ref s1, _), UseSegmentKind::Ident(ref s2, _)) => s1 == s2,
            (UseSegmentKind::Slf(_), UseSegmentKind::Slf(_))
            | (UseSegmentKind::Super(_), UseSegmentKind::Super(_))
            | (UseSegmentKind::Crate(_), UseSegmentKind::Crate(_))
            | (UseSegmentKind::Glob, UseSegmentKind::Glob) => true,
            (UseSegmentKind::List(ref list1), UseSegmentKind::List(ref list2)) => list1 == list2,
            _ => false,
        }
    }

    fn get_alias(&self) -> Option<&str> {
        match &self.kind {
            UseSegmentKind::Ident(_, a)
            | UseSegmentKind::Slf(a)
            | UseSegmentKind::Super(a)
            | UseSegmentKind::Crate(a) => a.as_deref(),
            _ => None,
        }
    }

    fn from_path_segment(
        context: &RewriteContext<'_>,
        path_seg: &ast::PathSegment,
        modsep: bool,
    ) -> Option<UseSegment> {
        let name = rewrite_ident(context, path_seg.ident);
        if name.is_empty() {
            return None;
        }
        let kind = match name {
            "self" => UseSegmentKind::Slf(None),
            "super" => UseSegmentKind::Super(None),
            "crate" => UseSegmentKind::Crate(None),
            _ => {
                let mod_sep = if modsep { "::" } else { "" };
                UseSegmentKind::Ident(format!("{mod_sep}{name}"), None)
            }
        };

        Some(UseSegment {
            kind,
            style_edition: context.config.style_edition(),
        })
    }

    fn contains_comment(&self) -> bool {
        if let UseSegmentKind::List(list) = &self.kind {
            list.iter().any(|subtree| subtree.contains_comment())
        } else {
            false
        }
    }
}

pub(crate) fn normalize_use_trees_with_granularity(
    use_trees: Vec<UseTree>,
    import_granularity: ImportGranularity,
) -> Vec<UseTree> {
    let merge_by = match import_granularity {
        ImportGranularity::Item => return flatten_use_trees(use_trees, ImportGranularity::Item),
        ImportGranularity::Preserve => return use_trees,
        ImportGranularity::Crate => SharedPrefix::Crate,
        ImportGranularity::Module => SharedPrefix::Module,
        ImportGranularity::One => SharedPrefix::One,
    };

    let mut result = Vec::with_capacity(use_trees.len());
    for use_tree in use_trees {
        if use_tree.contains_comment() || use_tree.attrs.is_some() {
            result.push(use_tree);
            continue;
        }

        for mut flattened in use_tree.flatten(import_granularity) {
            if let Some(tree) = result
                .iter_mut()
                .find(|tree| tree.share_prefix(&flattened, merge_by))
            {
                tree.merge(&flattened, merge_by);
            } else {
                // If this is the first tree with this prefix, handle potential trailing ::self
                if merge_by == SharedPrefix::Module {
                    flattened = flattened.nest_trailing_self();
                }
                result.push(flattened);
            }
        }
    }
    result
}

fn flatten_use_trees(
    use_trees: Vec<UseTree>,
    import_granularity: ImportGranularity,
) -> Vec<UseTree> {
    // Return non-sorted single occurrence of the use-trees text string;
    // order is by first occurrence of the use-tree.
    use_trees
        .into_iter()
        .flat_map(|tree| tree.flatten(import_granularity))
        .map(UseTree::nest_trailing_self)
        .unique()
        .collect()
}

impl fmt::Debug for UseTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Debug for UseSegment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.kind, f)
    }
}

impl fmt::Display for UseSegment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.kind, f)
    }
}

impl Hash for UseSegment {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
    }
}

impl fmt::Debug for UseSegmentKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for UseSegmentKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            UseSegmentKind::Glob => write!(f, "*"),
            UseSegmentKind::Ident(ref s, Some(ref alias)) => write!(f, "{s} as {alias}"),
            UseSegmentKind::Ident(ref s, None) => write!(f, "{s}"),
            UseSegmentKind::Slf(..) => write!(f, "self"),
            UseSegmentKind::Super(..) => write!(f, "super"),
            UseSegmentKind::Crate(..) => write!(f, "crate"),
            UseSegmentKind::List(ref list) => {
                write!(f, "{{")?;
                for (i, item) in list.iter().enumerate() {
                    if i != 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{item}")?;
                }
                write!(f, "}}")
            }
        }
    }
}
impl fmt::Display for UseTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, segment) in self.path.iter().enumerate() {
            if i != 0 {
                write!(f, "::")?;
            }
            write!(f, "{segment}")?;
        }
        Ok(())
    }
}

impl UseTree {
    // Rewrite use tree with `use ` and a trailing `;`.
    pub(crate) fn rewrite_top_level(
        &self,
        context: &RewriteContext<'_>,
        shape: Shape,
    ) -> RewriteResult {
        let vis = self.visibility.as_ref().map_or(Cow::from(""), |vis| {
            crate::utils::format_visibility(context, vis)
        });
        let use_str = self
            .rewrite_result(
                context,
                shape
                    .offset_left(vis.len())
                    .max_width_error(shape.width, self.span())?,
            )
            .map(|s| {
                if s.is_empty() {
                    s
                } else {
                    format!("{}use {};", vis, s)
                }
            })?;
        match self.attrs {
            Some(ref attrs) if !attrs.is_empty() => {
                let attr_str = attrs.rewrite_result(context, shape)?;
                let lo = attrs.last().unknown_error()?.span.hi();
                let hi = self.span.lo();
                let span = mk_sp(lo, hi);

                let allow_extend = if attrs.len() == 1 {
                    let line_len = attr_str.len() + 1 + use_str.len();
                    !attrs.first().unwrap().is_doc_comment()
                        && context.config.inline_attribute_width() >= line_len
                } else {
                    false
                };

                combine_strs_with_missing_comments(
                    context,
                    &attr_str,
                    &use_str,
                    span,
                    shape,
                    allow_extend,
                )
            }
            _ => Ok(use_str),
        }
    }

    // FIXME: Use correct span?
    // The given span is essentially incorrect, since we are reconstructing
    // use-statements. This should not be a problem, though, since we have
    // already tried to extract comment and observed that there are no comment
    // around the given use item, and the span will not be used afterward.
    fn from_path(path: Vec<UseSegment>, span: Span) -> UseTree {
        UseTree {
            path,
            span,
            list_item: None,
            visibility: None,
            attrs: None,
        }
    }

    pub(crate) fn from_ast_with_normalization(
        context: &RewriteContext<'_>,
        item: &ast::Item,
    ) -> Option<UseTree> {
        match item.kind {
            ast::ItemKind::Use(ref use_tree) => Some(
                UseTree::from_ast(
                    context,
                    use_tree,
                    None,
                    Some(item.vis.clone()),
                    Some(item.span.lo()),
                    if item.attrs.is_empty() {
                        None
                    } else {
                        Some(item.attrs.clone())
                    },
                )
                .normalize(),
            ),
            _ => None,
        }
    }

    fn from_ast(
        context: &RewriteContext<'_>,
        a: &ast::UseTree,
        list_item: Option<ListItem>,
        visibility: Option<ast::Visibility>,
        opt_lo: Option<BytePos>,
        attrs: Option<ast::AttrVec>,
    ) -> UseTree {
        let span = if let Some(lo) = opt_lo {
            mk_sp(lo, a.span.hi())
        } else {
            a.span
        };
        let mut result = UseTree {
            path: vec![],
            span,
            list_item,
            visibility,
            attrs,
        };

        let leading_modsep =
            context.config.edition() >= Edition::Edition2018 && a.prefix.is_global();

        let mut modsep = leading_modsep;

        for p in &a.prefix.segments {
            if let Some(use_segment) = UseSegment::from_path_segment(context, p, modsep) {
                result.path.push(use_segment);
                modsep = false;
            }
        }

        let style_edition = context.config.style_edition();

        match a.kind {
            UseTreeKind::Glob => {
                // in case of a global path and the glob starts at the root, e.g., "::*"
                if a.prefix.segments.len() == 1 && leading_modsep {
                    let kind = UseSegmentKind::Ident("".to_owned(), None);
                    result.path.push(UseSegment {
                        kind,
                        style_edition,
                    });
                }
                result.path.push(UseSegment {
                    kind: UseSegmentKind::Glob,
                    style_edition,
                });
            }
            UseTreeKind::Nested {
                items: ref list, ..
            } => {
                // Extract comments between nested use items.
                // This needs to be done before sorting use items.
                let items = itemize_list(
                    context.snippet_provider,
                    list.iter().map(|(tree, _)| tree),
                    "}",
                    ",",
                    |tree| tree.span.lo(),
                    |tree| tree.span.hi(),
                    |_| Ok("".to_owned()), // We only need comments for now.
                    context.snippet_provider.span_after(a.span, "{"),
                    a.span.hi(),
                    false,
                );

                // in case of a global path and the nested list starts at the root,
                // e.g., "::{foo, bar}"
                if a.prefix.segments.len() == 1 && leading_modsep {
                    let kind = UseSegmentKind::Ident("".to_owned(), None);
                    result.path.push(UseSegment {
                        kind,
                        style_edition,
                    });
                }
                let kind = UseSegmentKind::List(
                    list.iter()
                        .zip(items)
                        .map(|(t, list_item)| {
                            Self::from_ast(context, &t.0, Some(list_item), None, None, None)
                        })
                        .collect(),
                );
                result.path.push(UseSegment {
                    kind,
                    style_edition,
                });
            }
            UseTreeKind::Simple(ref rename) => {
                // If the path has leading double colons and is composed of only 2 segments, then we
                // bypass the call to path_to_imported_ident which would get only the ident and
                // lose the path root, e.g., `that` in `::that`.
                // The span of `a.prefix` contains the leading colons.
                let name = if a.prefix.segments.len() == 2 && leading_modsep {
                    context.snippet(a.prefix.span).to_owned()
                } else {
                    rewrite_ident(context, path_to_imported_ident(&a.prefix)).to_owned()
                };
                let alias = rename.and_then(|ident| {
                    if ident.name == sym::underscore_imports {
                        // for impl-only-use
                        Some("_".to_owned())
                    } else if ident == path_to_imported_ident(&a.prefix) {
                        None
                    } else {
                        Some(rewrite_ident(context, ident).to_owned())
                    }
                });
                let kind = match name.as_ref() {
                    "self" => UseSegmentKind::Slf(alias),
                    "super" => UseSegmentKind::Super(alias),
                    "crate" => UseSegmentKind::Crate(alias),
                    _ => UseSegmentKind::Ident(name, alias),
                };

                let segment = UseSegment {
                    kind,
                    style_edition,
                };

                // `name` is already in result.
                result.path.pop();
                result.path.push(segment);
            }
        }
        result
    }

    // Do the adjustments that rustfmt does elsewhere to use paths.
    pub(crate) fn normalize(mut self) -> UseTree {
        let mut last = self.path.pop().expect("Empty use tree?");
        // Hack around borrow checker.
        let mut normalize_sole_list = false;
        let mut aliased_self = false;

        // Remove foo::{} or self without attributes.
        match last.kind {
            _ if self.attrs.is_some() => (),
            UseSegmentKind::List(ref list) if list.is_empty() => {
                self.path = vec![];
                return self;
            }
            UseSegmentKind::Slf(None) if self.path.is_empty() && self.visibility.is_some() => {
                self.path = vec![];
                return self;
            }
            _ => (),
        }

        // Normalise foo::self -> foo.
        if let UseSegmentKind::Slf(None) = last.kind {
            if !self.path.is_empty() {
                return self;
            }
        }

        // Normalise foo::self as bar -> foo as bar.
        if let UseSegmentKind::Slf(_) = last.kind {
            if let Some(UseSegment {
                kind: UseSegmentKind::Ident(_, None),
                ..
            }) = self.path.last()
            {
                aliased_self = true;
            }
        }

        let mut done = false;
        if aliased_self {
            match self.path.last_mut() {
                Some(UseSegment {
                    kind: UseSegmentKind::Ident(_, ref mut old_rename),
                    ..
                }) => {
                    assert!(old_rename.is_none());
                    if let UseSegmentKind::Slf(Some(rename)) = last.clone().kind {
                        *old_rename = Some(rename);
                        done = true;
                    }
                }
                _ => unreachable!(),
            }
        }

        if done {
            return self;
        }

        // Normalise foo::{bar} -> foo::bar
        if let UseSegmentKind::List(ref list) = last.kind {
            if list.len() == 1 && list[0].to_string() != "self" && !list[0].has_comment() {
                normalize_sole_list = true;
            }
        }

        if normalize_sole_list {
            match last.kind {
                UseSegmentKind::List(list) => {
                    for seg in &list[0].path {
                        self.path.push(seg.clone());
                    }
                    return self.normalize();
                }
                _ => unreachable!(),
            }
        }

        // Recursively normalize elements of a list use (including sorting the list).
        if let UseSegmentKind::List(list) = last.kind {
            let mut list = list.into_iter().map(UseTree::normalize).collect::<Vec<_>>();
            list.sort();
            last = UseSegment {
                kind: UseSegmentKind::List(list),
                style_edition: last.style_edition,
            };
        }

        self.path.push(last);
        self
    }

    fn has_comment(&self) -> bool {
        self.list_item.as_ref().map_or(false, ListItem::has_comment)
    }

    fn contains_comment(&self) -> bool {
        self.has_comment() || self.path.iter().any(|path| path.contains_comment())
    }

    fn same_visibility(&self, other: &UseTree) -> bool {
        match (&self.visibility, &other.visibility) {
            (
                Some(ast::Visibility {
                    kind: ast::VisibilityKind::Inherited,
                    ..
                }),
                None,
            )
            | (
                None,
                Some(ast::Visibility {
                    kind: ast::VisibilityKind::Inherited,
                    ..
                }),
            )
            | (None, None) => true,
            (Some(ref a), Some(ref b)) => is_same_visibility(a, b),
            _ => false,
        }
    }

    fn share_prefix(&self, other: &UseTree, shared_prefix: SharedPrefix) -> bool {
        if self.path.is_empty()
            || other.path.is_empty()
            || self.attrs.is_some()
            || self.contains_comment()
            || !self.same_visibility(other)
        {
            false
        } else {
            match shared_prefix {
                SharedPrefix::Crate => self.path[0] == other.path[0],
                SharedPrefix::Module => {
                    self.path[..self.path.len() - 1] == other.path[..other.path.len() - 1]
                }
                SharedPrefix::One => true,
            }
        }
    }

    fn flatten(self, import_granularity: ImportGranularity) -> Vec<UseTree> {
        if self.path.is_empty() || self.contains_comment() {
            return vec![self];
        }
        match &self.path.clone().last().unwrap().kind {
            UseSegmentKind::List(list) => {
                if list.len() == 1 && list[0].path.len() == 1 {
                    if let UseSegmentKind::Slf(..) = list[0].path[0].kind {
                        return vec![self];
                    };
                }
                let prefix = &self.path[..self.path.len() - 1];
                let mut result = vec![];
                for nested_use_tree in list {
                    for flattened in &mut nested_use_tree.clone().flatten(import_granularity) {
                        let mut new_path = prefix.to_vec();
                        new_path.append(&mut flattened.path);
                        result.push(UseTree {
                            path: new_path,
                            span: self.span,
                            list_item: None,
                            visibility: self.visibility.clone(),
                            // only retain attributes for `ImportGranularity::Item`
                            attrs: match import_granularity {
                                ImportGranularity::Item => self.attrs.clone(),
                                _ => None,
                            },
                        });
                    }
                }

                result
            }
            _ => vec![self],
        }
    }

    fn merge(&mut self, other: &UseTree, merge_by: SharedPrefix) {
        let mut prefix = 0;
        for (a, b) in self.path.iter().zip(other.path.iter()) {
            // only discard the alias at the root of the tree
            if (prefix == 0 && a.equal_except_alias(b)) || a == b {
                prefix += 1;
            } else {
                break;
            }
        }
        if let Some(new_path) = merge_rest(&self.path, &other.path, prefix, merge_by) {
            self.path = new_path;
            self.span = self.span.to(other.span);
        }
    }

    /// If this tree ends in `::self`, rewrite it to `::{self}`.
    fn nest_trailing_self(mut self) -> UseTree {
        if let Some(UseSegment {
            kind: UseSegmentKind::Slf(..),
            ..
        }) = self.path.last()
        {
            let self_segment = self.path.pop().unwrap();
            let style_edition = self_segment.style_edition;
            let kind = UseSegmentKind::List(vec![UseTree::from_path(vec![self_segment], DUMMY_SP)]);
            self.path.push(UseSegment {
                kind,
                style_edition,
            });
        }
        self
    }
}

fn merge_rest(
    a: &[UseSegment],
    b: &[UseSegment],
    mut len: usize,
    merge_by: SharedPrefix,
) -> Option<Vec<UseSegment>> {
    if a.len() == len && b.len() == len {
        return None;
    }
    if a.len() != len && b.len() != len {
        let style_edition = a[len].style_edition;
        if let UseSegmentKind::List(ref list) = a[len].kind {
            let mut list = list.clone();
            merge_use_trees_inner(
                &mut list,
                UseTree::from_path(b[len..].to_vec(), DUMMY_SP),
                merge_by,
            );
            let mut new_path = b[..len].to_vec();
            let kind = UseSegmentKind::List(list);
            new_path.push(UseSegment {
                kind,
                style_edition,
            });
            return Some(new_path);
        }
    } else if len == 1 {
        let (common, rest) = if a.len() == len {
            (&a[0], &b[1..])
        } else {
            (&b[0], &a[1..])
        };
        let kind = UseSegmentKind::Slf(common.get_alias().map(ToString::to_string));
        let style_edition = a[0].style_edition;
        let mut list = vec![UseTree::from_path(
            vec![UseSegment {
                kind,
                style_edition,
            }],
            DUMMY_SP,
        )];
        match rest {
            [
                UseSegment {
                    kind: UseSegmentKind::List(rest_list),
                    ..
                },
            ] => list.extend(rest_list.clone()),
            _ => list.push(UseTree::from_path(rest.to_vec(), DUMMY_SP)),
        }
        return Some(vec![
            b[0].clone(),
            UseSegment {
                kind: UseSegmentKind::List(list),
                style_edition,
            },
        ]);
    } else {
        len -= 1;
    }
    let mut list = vec![
        UseTree::from_path(a[len..].to_vec(), DUMMY_SP),
        UseTree::from_path(b[len..].to_vec(), DUMMY_SP),
    ];
    list.sort();
    let mut new_path = b[..len].to_vec();
    let kind = UseSegmentKind::List(list);
    let style_edition = a[0].style_edition;
    new_path.push(UseSegment {
        kind,
        style_edition,
    });
    Some(new_path)
}

fn merge_use_trees_inner(trees: &mut Vec<UseTree>, use_tree: UseTree, merge_by: SharedPrefix) {
    struct SimilarTree<'a> {
        similarity: usize,
        path_len: usize,
        tree: &'a mut UseTree,
    }

    let similar_trees = trees.iter_mut().filter_map(|tree| {
        if tree.share_prefix(&use_tree, merge_by) {
            // In the case of `SharedPrefix::One`, `similarity` is used for deciding with which
            // tree `use_tree` should be merge.
            // In other cases `similarity` won't be used, so set it to `0` as a dummy value.
            let similarity = if merge_by == SharedPrefix::One {
                tree.path
                    .iter()
                    .zip(&use_tree.path)
                    .take_while(|(a, b)| a.equal_except_alias(b))
                    .count()
            } else {
                0
            };

            let path_len = tree.path.len();
            Some(SimilarTree {
                similarity,
                tree,
                path_len,
            })
        } else {
            None
        }
    });

    if use_tree.path.len() == 1 && merge_by == SharedPrefix::Crate {
        if let Some(tree) = similar_trees.min_by_key(|tree| tree.path_len) {
            if tree.path_len == 1 {
                return;
            }
        }
    } else if merge_by == SharedPrefix::One {
        if let Some(sim_tree) = similar_trees.max_by_key(|tree| tree.similarity) {
            if sim_tree.similarity > 0 {
                sim_tree.tree.merge(&use_tree, merge_by);
                return;
            }
        }
    } else if let Some(sim_tree) = similar_trees.max_by_key(|tree| tree.path_len) {
        if sim_tree.path_len > 1 {
            sim_tree.tree.merge(&use_tree, merge_by);
            return;
        }
    }
    trees.push(use_tree);
    trees.sort();
}

impl Hash for UseTree {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.path.hash(state);
    }
}

impl PartialOrd for UseSegment {
    fn partial_cmp(&self, other: &UseSegment) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialOrd for UseTree {
    fn partial_cmp(&self, other: &UseTree) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for UseSegment {
    fn cmp(&self, other: &UseSegment) -> Ordering {
        use self::UseSegmentKind::*;

        fn is_upper_snake_case(s: &str) -> bool {
            s.chars()
                .all(|c| c.is_uppercase() || c == '_' || c.is_numeric())
        }

        match (&self.kind, &other.kind) {
            (Slf(ref a), Slf(ref b))
            | (Super(ref a), Super(ref b))
            | (Crate(ref a), Crate(ref b)) => match (a, b) {
                (Some(sa), Some(sb)) => {
                    if self.style_edition >= StyleEdition::Edition2024 {
                        version_sort(sa.trim_start_matches("r#"), sb.trim_start_matches("r#"))
                    } else {
                        a.cmp(b)
                    }
                }
                (_, _) => a.cmp(b),
            },
            (Glob, Glob) => Ordering::Equal,
            (Ident(ref pia, ref aa), Ident(ref pib, ref ab)) => {
                let (ia, ib) = if self.style_edition >= StyleEdition::Edition2024 {
                    (pia.trim_start_matches("r#"), pib.trim_start_matches("r#"))
                } else {
                    (pia.as_str(), pib.as_str())
                };

                let ident_ord = if self.style_edition >= StyleEdition::Edition2024 {
                    version_sort(ia, ib)
                } else {
                    // snake_case < CamelCase < UPPER_SNAKE_CASE
                    if ia.starts_with(char::is_uppercase) && ib.starts_with(char::is_lowercase) {
                        return Ordering::Greater;
                    }
                    if ia.starts_with(char::is_lowercase) && ib.starts_with(char::is_uppercase) {
                        return Ordering::Less;
                    }
                    if is_upper_snake_case(ia) && !is_upper_snake_case(ib) {
                        return Ordering::Greater;
                    }
                    if !is_upper_snake_case(ia) && is_upper_snake_case(ib) {
                        return Ordering::Less;
                    }
                    ia.cmp(ib)
                };

                if ident_ord != Ordering::Equal {
                    return ident_ord;
                }
                match (aa, ab) {
                    (None, Some(_)) => Ordering::Less,
                    (Some(_), None) => Ordering::Greater,
                    (Some(aas), Some(abs)) => {
                        if self.style_edition >= StyleEdition::Edition2024 {
                            version_sort(aas.trim_start_matches("r#"), abs.trim_start_matches("r#"))
                        } else {
                            aas.cmp(abs)
                        }
                    }
                    (None, None) => Ordering::Equal,
                }
            }
            (List(ref a), List(ref b)) => {
                for (a, b) in a.iter().zip(b.iter()) {
                    let ord = a.cmp(b);
                    if ord != Ordering::Equal {
                        return ord;
                    }
                }

                a.len().cmp(&b.len())
            }
            (Slf(_), _) => Ordering::Less,
            (_, Slf(_)) => Ordering::Greater,
            (Super(_), _) => Ordering::Less,
            (_, Super(_)) => Ordering::Greater,
            (Crate(_), _) => Ordering::Less,
            (_, Crate(_)) => Ordering::Greater,
            (Ident(..), _) => Ordering::Less,
            (_, Ident(..)) => Ordering::Greater,
            (Glob, _) => Ordering::Less,
            (_, Glob) => Ordering::Greater,
        }
    }
}
impl Ord for UseTree {
    fn cmp(&self, other: &UseTree) -> Ordering {
        for (a, b) in self.path.iter().zip(other.path.iter()) {
            let ord = a.cmp(b);
            // The comparison without aliases is a hack to avoid situations like
            // comparing `a::b` to `a as c` - where the latter should be ordered
            // first since it is shorter.
            if ord != Ordering::Equal && a.remove_alias().cmp(&b.remove_alias()) != Ordering::Equal
            {
                return ord;
            }
        }

        self.path.len().cmp(&other.path.len())
    }
}

fn rewrite_nested_use_tree(
    context: &RewriteContext<'_>,
    use_tree_list: &[UseTree],
    shape: Shape,
) -> RewriteResult {
    let mut list_items = Vec::with_capacity(use_tree_list.len());
    let nested_shape = match context.config.imports_indent() {
        IndentStyle::Block => shape
            .block_indent(context.config.tab_spaces())
            .with_max_width(context.config)
            .sub_width(1)
            .unknown_error()?,
        IndentStyle::Visual => shape.visual_indent(0),
    };
    for use_tree in use_tree_list {
        if let Some(mut list_item) = use_tree.list_item.clone() {
            list_item.item = use_tree.rewrite_result(context, nested_shape);
            list_items.push(list_item);
        } else {
            list_items.push(ListItem::from_str(
                use_tree.rewrite_result(context, nested_shape)?,
            ));
        }
    }
    let has_nested_list = use_tree_list.iter().any(|use_segment| {
        use_segment.path.last().map_or(false, |last_segment| {
            matches!(last_segment.kind, UseSegmentKind::List(..))
        })
    });

    let remaining_width = if has_nested_list {
        0
    } else {
        shape.width.saturating_sub(2)
    };

    let tactic = definitive_tactic(
        &list_items,
        context.config.imports_layout(),
        Separator::Comma,
        remaining_width,
    );

    let ends_with_newline = context.config.imports_indent() == IndentStyle::Block
        && tactic != DefinitiveListTactic::Horizontal;
    let trailing_separator = if ends_with_newline {
        context.config.trailing_comma()
    } else {
        SeparatorTactic::Never
    };
    let fmt = ListFormatting::new(nested_shape, context.config)
        .tactic(tactic)
        .trailing_separator(trailing_separator)
        .ends_with_newline(ends_with_newline)
        .preserve_newline(true)
        .nested(has_nested_list);

    let list_str = write_list(&list_items, &fmt)?;

    let result = if (list_str.contains('\n')
        || list_str.len() > remaining_width
        || tactic == DefinitiveListTactic::Vertical)
        && context.config.imports_indent() == IndentStyle::Block
    {
        format!(
            "{{\n{}{}\n{}}}",
            nested_shape.indent.to_string(context.config),
            list_str,
            shape.indent.to_string(context.config)
        )
    } else {
        format!("{{{list_str}}}")
    };

    Ok(result)
}

impl Rewrite for UseSegment {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        self.rewrite_result(context, shape).ok()
    }

    fn rewrite_result(&self, context: &RewriteContext<'_>, shape: Shape) -> RewriteResult {
        Ok(match self.kind {
            UseSegmentKind::Ident(ref ident, Some(ref rename)) => {
                format!("{ident} as {rename}")
            }
            UseSegmentKind::Ident(ref ident, None) => ident.clone(),
            UseSegmentKind::Slf(Some(ref rename)) => format!("self as {rename}"),
            UseSegmentKind::Slf(None) => "self".to_owned(),
            UseSegmentKind::Super(Some(ref rename)) => format!("super as {rename}"),
            UseSegmentKind::Super(None) => "super".to_owned(),
            UseSegmentKind::Crate(Some(ref rename)) => format!("crate as {rename}"),
            UseSegmentKind::Crate(None) => "crate".to_owned(),
            UseSegmentKind::Glob => "*".to_owned(),
            UseSegmentKind::List(ref use_tree_list) => {
                rewrite_nested_use_tree(
                    context,
                    use_tree_list,
                    // 1 = "{" and "}"
                    shape
                        .offset_left(1)
                        .and_then(|s| s.sub_width(1))
                        .unknown_error()?,
                )?
            }
        })
    }
}

impl Rewrite for UseTree {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        self.rewrite_result(context, shape).ok()
    }

    // This does NOT format attributes and visibility or add a trailing `;`.
    fn rewrite_result(&self, context: &RewriteContext<'_>, mut shape: Shape) -> RewriteResult {
        let mut result = String::with_capacity(256);
        let mut iter = self.path.iter().peekable();
        while let Some(segment) = iter.next() {
            let segment_str = segment.rewrite_result(context, shape)?;
            result.push_str(&segment_str);
            if iter.peek().is_some() {
                result.push_str("::");
                // 2 = "::"
                shape = shape
                    .offset_left(2 + segment_str.len())
                    .max_width_error(shape.width, self.span())?;
            }
        }
        Ok(result)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum SharedPrefix {
    Crate,
    Module,
    One,
}

#[cfg(test)]
mod test {
    use super::*;

    // Parse the path part of an import. This parser is not robust and is only
    // suitable for use in a test harness.
    fn parse_use_tree(s: &str) -> UseTree {
        use std::iter::Peekable;
        use std::mem::swap;
        use std::str::Chars;

        struct Parser<'a> {
            input: Peekable<Chars<'a>>,
            style_edition: StyleEdition,
        }

        impl<'a> Parser<'a> {
            fn bump(&mut self) {
                self.input.next().unwrap();
            }

            fn eat(&mut self, c: char) {
                assert_eq!(self.input.next().unwrap(), c);
            }

            fn push_segment(
                &self,
                result: &mut Vec<UseSegment>,
                buf: &mut String,
                alias_buf: &mut Option<String>,
            ) {
                let style_edition = self.style_edition;
                if !buf.is_empty() {
                    let mut alias = None;
                    swap(alias_buf, &mut alias);

                    match buf.as_ref() {
                        "self" => {
                            let kind = UseSegmentKind::Slf(alias);
                            result.push(UseSegment {
                                kind,
                                style_edition,
                            });
                            *buf = String::new();
                            *alias_buf = None;
                        }
                        "super" => {
                            let kind = UseSegmentKind::Super(alias);
                            result.push(UseSegment {
                                kind,
                                style_edition,
                            });
                            *buf = String::new();
                            *alias_buf = None;
                        }
                        "crate" => {
                            let kind = UseSegmentKind::Crate(alias);
                            result.push(UseSegment {
                                kind,
                                style_edition,
                            });
                            *buf = String::new();
                            *alias_buf = None;
                        }
                        _ => {
                            let mut name = String::new();
                            swap(buf, &mut name);
                            let kind = UseSegmentKind::Ident(name, alias);
                            result.push(UseSegment {
                                kind,
                                style_edition,
                            });
                        }
                    }
                }
            }

            fn parse_in_list(&mut self) -> UseTree {
                let mut result = vec![];
                let mut buf = String::new();
                let mut alias_buf = None;
                while let Some(&c) = self.input.peek() {
                    match c {
                        '{' => {
                            assert!(buf.is_empty());
                            self.bump();
                            let kind = UseSegmentKind::List(self.parse_list());
                            result.push(UseSegment {
                                kind,
                                style_edition: self.style_edition,
                            });
                            self.eat('}');
                        }
                        '*' => {
                            assert!(buf.is_empty());
                            self.bump();
                            let kind = UseSegmentKind::Glob;
                            result.push(UseSegment {
                                kind,
                                style_edition: self.style_edition,
                            });
                        }
                        ':' => {
                            self.bump();
                            self.eat(':');
                            self.push_segment(&mut result, &mut buf, &mut alias_buf);
                        }
                        '}' | ',' => {
                            self.push_segment(&mut result, &mut buf, &mut alias_buf);
                            return UseTree {
                                path: result,
                                span: DUMMY_SP,
                                list_item: None,
                                visibility: None,
                                attrs: None,
                            };
                        }
                        ' ' => {
                            self.bump();
                            self.eat('a');
                            self.eat('s');
                            self.eat(' ');
                            alias_buf = Some(String::new());
                        }
                        c => {
                            self.bump();
                            if let Some(ref mut buf) = alias_buf {
                                buf.push(c);
                            } else {
                                buf.push(c);
                            }
                        }
                    }
                }
                self.push_segment(&mut result, &mut buf, &mut alias_buf);
                UseTree {
                    path: result,
                    span: DUMMY_SP,
                    list_item: None,
                    visibility: None,
                    attrs: None,
                }
            }

            fn parse_list(&mut self) -> Vec<UseTree> {
                let mut result = vec![];
                loop {
                    match self.input.peek().unwrap() {
                        ',' | ' ' => self.bump(),
                        '}' => {
                            return result;
                        }
                        _ => result.push(self.parse_in_list()),
                    }
                }
            }
        }

        let mut parser = Parser {
            input: s.chars().peekable(),
            style_edition: StyleEdition::Edition2015,
        };
        parser.parse_in_list()
    }

    macro_rules! parse_use_trees {
        ($($s:expr),* $(,)*) => {
            vec![
                $(parse_use_tree($s),)*
            ]
        }
    }

    macro_rules! test_merge {
        ($by:ident, [$($input:expr),* $(,)*], [$($output:expr),* $(,)*]) => {
            assert_eq!(
                normalize_use_trees_with_granularity(
                    parse_use_trees!($($input,)*),
                    ImportGranularity::$by,
                ),
                parse_use_trees!($($output,)*),
            );
        }
    }

    #[test]
    fn test_use_tree_merge_crate() {
        test_merge!(
            Crate,
            ["a::b::{c, d}", "a::b::{e, f}"],
            ["a::b::{c, d, e, f}"]
        );
        test_merge!(Crate, ["a::b::c", "a::b"], ["a::{b, b::c}"]);
        test_merge!(Crate, ["a::b", "a::b"], ["a::b"]);
        test_merge!(Crate, ["a", "a::b", "a::b::c"], ["a::{self, b, b::c}"]);
        test_merge!(
            Crate,
            ["a", "a::b", "a::b::c", "a::b::c::d"],
            ["a::{self, b, b::{c, c::d}}"]
        );
        test_merge!(
            Crate,
            ["a", "a::b", "a::b::c", "a::b"],
            ["a::{self, b, b::c}"]
        );
        test_merge!(
            Crate,
            ["a::{b::{self, c}, d::e}", "a::d::f"],
            ["a::{b::{self, c}, d::{e, f}}"]
        );
        test_merge!(
            Crate,
            ["a::d::f", "a::{b::{self, c}, d::e}"],
            ["a::{b::{self, c}, d::{e, f}}"]
        );
        test_merge!(
            Crate,
            ["a::{c, d, b}", "a::{d, e, b, a, f}", "a::{f, g, c}"],
            ["a::{a, b, c, d, e, f, g}"]
        );
        test_merge!(
            Crate,
            ["a::{self}", "b::{self as foo}"],
            ["a::{self}", "b::{self as foo}"]
        );
    }

    #[test]
    fn test_use_tree_merge_module() {
        test_merge!(
            Module,
            ["foo::b", "foo::{a, c, d::e}"],
            ["foo::{a, b, c}", "foo::d::e"]
        );

        test_merge!(
            Module,
            ["foo::{a::b, a::c, d::e, d::f}"],
            ["foo::a::{b, c}", "foo::d::{e, f}"]
        );
    }

    #[test]
    fn test_use_tree_merge_one() {
        test_merge!(One, ["a", "b"], ["{a, b}"]);

        test_merge!(One, ["a::{aa, ab}", "b", "a"], ["{a::{self, aa, ab}, b}"]);

        test_merge!(One, ["a as x", "b as y"], ["{a as x, b as y}"]);

        test_merge!(
            One,
            ["a::{aa as xa, ab}", "b", "a"],
            ["{a::{self, aa as xa, ab}, b}"]
        );

        test_merge!(
            One,
            ["a", "a::{aa, ab::{aba, abb}}"],
            ["a::{self, aa, ab::{aba, abb}}"]
        );

        test_merge!(One, ["a", "b::{ba, *}"], ["{a, b::{ba, *}}"]);

        test_merge!(One, ["a", "b", "a::aa"], ["{a::{self, aa}, b}"]);

        test_merge!(
            One,
            ["a::aa::aaa", "a::ac::aca", "a::aa::*"],
            ["a::{aa::{aaa, *}, ac::aca}"]
        );

        test_merge!(
            One,
            ["a", "b::{ba, bb}", "a::{aa::*, ab::aba}"],
            ["{a::{self, aa::*, ab::aba}, b::{ba, bb}}"]
        );

        test_merge!(
            One,
            ["b", "a::ac::{aca, acb}", "a::{aa::*, ab}"],
            ["{a::{aa::*, ab, ac::{aca, acb}}, b}"]
        );
    }

    #[test]
    fn test_flatten_use_trees() {
        assert_eq!(
            flatten_use_trees(
                parse_use_trees!["foo::{a::{b, c}, d::e}"],
                ImportGranularity::Item
            ),
            parse_use_trees!["foo::a::b", "foo::a::c", "foo::d::e"]
        );

        assert_eq!(
            flatten_use_trees(
                parse_use_trees!["foo::{self, a, b::{c, d}, e::*}"],
                ImportGranularity::Item
            ),
            parse_use_trees![
                "foo::{self}",
                "foo::a",
                "foo::b::c",
                "foo::b::d",
                "foo::e::*"
            ]
        );
    }

    #[test]
    fn test_use_tree_flatten() {
        assert_eq!(
            parse_use_tree("a::b::{c, d, e, f}").flatten(ImportGranularity::Item),
            parse_use_trees!("a::b::c", "a::b::d", "a::b::e", "a::b::f",)
        );

        assert_eq!(
            parse_use_tree("a::b::{c::{d, e, f}, g, h::{i, j, k}}")
                .flatten(ImportGranularity::Item),
            parse_use_trees![
                "a::b::c::d",
                "a::b::c::e",
                "a::b::c::f",
                "a::b::g",
                "a::b::h::i",
                "a::b::h::j",
                "a::b::h::k",
            ]
        );
    }

    #[test]
    fn test_use_tree_normalize() {
        assert_eq!(parse_use_tree("a::self").normalize(), parse_use_tree("a"));
        assert_eq!(
            parse_use_tree("a::self as foo").normalize(),
            parse_use_tree("a as foo")
        );
        assert_eq!(
            parse_use_tree("a::{self}").normalize(),
            parse_use_tree("a::{self}")
        );
        assert_eq!(parse_use_tree("a::{b}").normalize(), parse_use_tree("a::b"));
        assert_eq!(
            parse_use_tree("a::{b, c::self}").normalize(),
            parse_use_tree("a::{b, c}")
        );
        assert_eq!(
            parse_use_tree("a::{b as bar, c::self}").normalize(),
            parse_use_tree("a::{b as bar, c}")
        );
    }

    #[test]
    fn test_use_tree_ord() {
        assert!(parse_use_tree("a").normalize() < parse_use_tree("aa").normalize());
        assert!(parse_use_tree("a").normalize() < parse_use_tree("a::a").normalize());
        assert!(parse_use_tree("a").normalize() < parse_use_tree("*").normalize());
        assert!(parse_use_tree("a").normalize() < parse_use_tree("{a, b}").normalize());
        assert!(parse_use_tree("*").normalize() < parse_use_tree("{a, b}").normalize());

        assert!(
            parse_use_tree("aaaaaaaaaaaaaaa::{bb, cc, dddddddd}").normalize()
                < parse_use_tree("aaaaaaaaaaaaaaa::{bb, cc, ddddddddd}").normalize()
        );
        assert!(
            parse_use_tree("serde::de::{Deserialize}").normalize()
                < parse_use_tree("serde_json").normalize()
        );
        assert!(parse_use_tree("a::b::c").normalize() < parse_use_tree("a::b::*").normalize());
        assert!(
            parse_use_tree("foo::{Bar, Baz}").normalize()
                < parse_use_tree("{Bar, Baz}").normalize()
        );

        assert!(
            parse_use_tree("foo::{qux as bar}").normalize()
                < parse_use_tree("foo::{self as bar}").normalize()
        );
        assert!(
            parse_use_tree("foo::{qux as bar}").normalize()
                < parse_use_tree("foo::{baz, qux as bar}").normalize()
        );
        assert!(
            parse_use_tree("foo::{self as bar, baz}").normalize()
                < parse_use_tree("foo::{baz, qux as bar}").normalize()
        );

        assert!(parse_use_tree("foo").normalize() < parse_use_tree("Foo").normalize());
        assert!(parse_use_tree("foo").normalize() < parse_use_tree("foo::Bar").normalize());

        assert!(
            parse_use_tree("std::cmp::{d, c, b, a}").normalize()
                < parse_use_tree("std::cmp::{b, e, g, f}").normalize()
        );
    }

    #[test]
    fn test_use_tree_nest_trailing_self() {
        assert_eq!(
            parse_use_tree("a::b::self").nest_trailing_self(),
            parse_use_tree("a::b::{self}")
        );
        assert_eq!(
            parse_use_tree("a::b::c").nest_trailing_self(),
            parse_use_tree("a::b::c")
        );
        assert_eq!(
            parse_use_tree("a::b::{c, d}").nest_trailing_self(),
            parse_use_tree("a::b::{c, d}")
        );
        assert_eq!(
            parse_use_tree("a::b::{self, c}").nest_trailing_self(),
            parse_use_tree("a::b::{self, c}")
        );
    }
}
