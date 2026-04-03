//! Documentation extraction and source mapping.
//!
//! This module handles the extraction and processing of doc comments and `#[doc = "..."]`
//! attributes, including macro expansion for `#[doc = macro!()]` patterns.
//! It builds a concatenated string of the full docs as well as a source map
//! to map it back to AST (which is needed for things like resolving links in doc comments
//! and highlight injection).

use std::{
    convert::Infallible,
    ops::{ControlFlow, Range},
};

use base_db::Crate;
use cfg::CfgOptions;
use either::Either;
use hir_expand::{
    AstId, ExpandTo, HirFileId, InFile,
    attrs::{Meta, expand_cfg_attr_with_doc_comments},
    mod_path::ModPath,
    span_map::SpanMap,
};
use span::AstIdMap;
use syntax::{
    AstNode, AstToken, SyntaxNode,
    ast::{self, AttrDocCommentIter, HasAttrs, IsString},
};
use tt::{TextRange, TextSize};

use crate::{db::DefDatabase, macro_call_as_call_id, nameres::MacroSubNs, resolver::Resolver};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct DocsSourceMapLine {
    /// The offset in [`Docs::docs`].
    string_offset: TextSize,
    /// The offset in the AST of the text. `None` for macro-expanded doc strings
    /// where we cannot provide a faithful source mapping.
    ast_offset: Option<TextSize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Docs {
    /// The concatenated string of all `#[doc = "..."]` attributes and documentation comments.
    docs: String,
    /// A sorted map from an offset in `docs` to an offset in the source code.
    docs_source_map: Vec<DocsSourceMapLine>,
    /// If the item is an outlined module (`mod foo;`), `docs_source_map` store the concatenated
    /// list of the outline and inline docs (outline first). Then, this field contains the [`HirFileId`]
    /// of the outline declaration, and the index in `docs` from which the inline docs
    /// begin.
    outline_mod: Option<(HirFileId, usize)>,
    inline_file: HirFileId,
    /// The size the prepended prefix, which does not map to real doc comments.
    prefix_len: TextSize,
    /// The offset in `docs` from which the docs are inner attributes/comments.
    inline_inner_docs_start: Option<TextSize>,
    /// Like `inline_inner_docs_start`, but for `outline_mod`. This can happen only when merging `Docs`
    /// (as outline modules don't have inner attributes).
    outline_inner_docs_start: Option<TextSize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsInnerDoc {
    No,
    Yes,
}

impl IsInnerDoc {
    #[inline]
    pub fn yes(self) -> bool {
        self == IsInnerDoc::Yes
    }
}

impl Docs {
    #[inline]
    pub fn docs(&self) -> &str {
        &self.docs
    }

    #[inline]
    pub fn into_docs(self) -> String {
        self.docs
    }

    pub fn find_ast_range(
        &self,
        mut string_range: TextRange,
    ) -> Option<(InFile<TextRange>, IsInnerDoc)> {
        if string_range.start() < self.prefix_len {
            return None;
        }
        string_range -= self.prefix_len;

        let mut file = self.inline_file;
        let mut inner_docs_start = self.inline_inner_docs_start;
        // Check whether the range is from the outline, the inline, or both.
        let source_map = if let Some((outline_mod_file, outline_mod_end)) = self.outline_mod {
            if let Some(first_inline) = self.docs_source_map.get(outline_mod_end) {
                if string_range.end() <= first_inline.string_offset {
                    // The range is completely in the outline.
                    file = outline_mod_file;
                    inner_docs_start = self.outline_inner_docs_start;
                    &self.docs_source_map[..outline_mod_end]
                } else if string_range.start() >= first_inline.string_offset {
                    // The range is completely in the inline.
                    &self.docs_source_map[outline_mod_end..]
                } else {
                    // The range is combined from the outline and the inline - cannot map it back.
                    return None;
                }
            } else {
                // There is no inline.
                file = outline_mod_file;
                inner_docs_start = self.outline_inner_docs_start;
                &self.docs_source_map
            }
        } else {
            // There is no outline.
            &self.docs_source_map
        };

        let after_range =
            source_map.partition_point(|line| line.string_offset <= string_range.start()) - 1;
        let after_range = &source_map[after_range..];
        let line = after_range.first()?;
        // Unmapped lines (from macro-expanded docs) cannot be mapped back to AST.
        let ast_offset = line.ast_offset?;
        if after_range.get(1).is_some_and(|next_line| next_line.string_offset < string_range.end())
        {
            // The range is combined from two lines - cannot map it back.
            return None;
        }
        let ast_range = string_range - line.string_offset + ast_offset;
        let is_inner = if inner_docs_start
            .is_some_and(|inner_docs_start| string_range.start() >= inner_docs_start)
        {
            IsInnerDoc::Yes
        } else {
            IsInnerDoc::No
        };
        Some((InFile::new(file, ast_range), is_inner))
    }

    #[inline]
    pub fn shift_by(&mut self, offset: TextSize) {
        self.prefix_len += offset;
    }

    pub fn prepend_str(&mut self, s: &str) {
        self.prefix_len += TextSize::of(s);
        self.docs.insert_str(0, s);
    }

    pub fn append_str(&mut self, s: &str) {
        self.docs.push_str(s);
    }

    pub fn append(&mut self, other: &Docs) {
        let other_offset = TextSize::of(&self.docs);

        assert!(
            self.outline_mod.is_none() && other.outline_mod.is_none(),
            "cannot merge `Docs` that have `outline_mod` set"
        );
        self.outline_mod = Some((self.inline_file, self.docs_source_map.len()));
        self.inline_file = other.inline_file;
        self.outline_inner_docs_start = self.inline_inner_docs_start;
        self.inline_inner_docs_start = other.inline_inner_docs_start.map(|it| it + other_offset);

        self.docs.push_str(&other.docs);
        self.docs_source_map.extend(other.docs_source_map.iter().map(
            |&DocsSourceMapLine { string_offset, ast_offset }| DocsSourceMapLine {
                ast_offset,
                string_offset: string_offset + other_offset,
            },
        ));
    }

    fn extend_with_doc_comment(&mut self, comment: ast::Comment, indent: &mut usize) {
        let Some((doc, offset)) = comment.doc_comment() else { return };
        self.extend_with_doc_str(doc, comment.syntax().text_range().start() + offset, indent);
    }

    fn extend_with_doc_attr(&mut self, value: syntax::SyntaxToken, indent: &mut usize) {
        let Some(value) = ast::String::cast(value) else { return };
        let Some(value_offset) = value.text_range_between_quotes() else { return };
        let value_offset = value_offset.start();
        let Ok(value) = value.value() else { return };
        // FIXME: Handle source maps for escaped text.
        self.extend_with_doc_str(&value, value_offset, indent);
    }

    pub(crate) fn extend_with_doc_str(
        &mut self,
        doc: &str,
        offset_in_ast: TextSize,
        indent: &mut usize,
    ) {
        self.push_doc_lines(doc, Some(offset_in_ast), indent);
    }

    fn extend_with_unmapped_doc_str(&mut self, doc: &str, indent: &mut usize) {
        self.push_doc_lines(doc, None, indent);
    }

    fn push_doc_lines(&mut self, doc: &str, mut ast_offset: Option<TextSize>, indent: &mut usize) {
        for line in doc.split('\n') {
            self.docs_source_map
                .push(DocsSourceMapLine { string_offset: TextSize::of(&self.docs), ast_offset });
            if let Some(ref mut offset) = ast_offset {
                *offset += TextSize::of(line) + TextSize::of("\n");
            }

            let line = line.trim_end();
            if let Some(line_indent) = line.chars().position(|ch| !ch.is_whitespace()) {
                // Empty lines are handled because `position()` returns `None` for them.
                *indent = std::cmp::min(*indent, line_indent);
            }
            self.docs.push_str(line);
            self.docs.push('\n');
        }
    }

    fn remove_indent(&mut self, indent: usize, start_source_map_index: usize) {
        /// In case of panics, we want to avoid corrupted UTF-8 in `self.docs`, so we clear it.
        struct Guard<'a>(&'a mut Docs);
        impl Drop for Guard<'_> {
            fn drop(&mut self) {
                let Docs {
                    docs,
                    docs_source_map,
                    outline_mod,
                    inline_file: _,
                    prefix_len: _,
                    inline_inner_docs_start: _,
                    outline_inner_docs_start: _,
                } = self.0;
                // Don't use `String::clear()` here because it's not guaranteed to not do UTF-8-dependent things,
                // and we may have temporarily broken the string's encoding.
                unsafe { docs.as_mut_vec() }.clear();
                // This is just to avoid panics down the road.
                docs_source_map.clear();
                *outline_mod = None;
            }
        }

        if self.docs.is_empty() {
            return;
        }

        let guard = Guard(self);
        let source_map = &mut guard.0.docs_source_map[start_source_map_index..];
        let Some(&DocsSourceMapLine { string_offset: mut copy_into, .. }) = source_map.first()
        else {
            return;
        };
        // We basically want to remove multiple ranges from a string. Doing this efficiently (without O(N^2)
        // or allocations) requires unsafe. Basically, for each line, we copy the line minus the indent into
        // consecutive to the previous line (which may have moved). Then at the end we truncate.
        let mut accumulated_offset = TextSize::new(0);
        for idx in 0..source_map.len() {
            let string_end_offset = source_map
                .get(idx + 1)
                .map_or_else(|| TextSize::of(&guard.0.docs), |next_attr| next_attr.string_offset);
            let line_source = &mut source_map[idx];
            let line_docs =
                &guard.0.docs[TextRange::new(line_source.string_offset, string_end_offset)];
            let line_docs_len = TextSize::of(line_docs);
            let indent_size = line_docs.char_indices().nth(indent).map_or_else(
                || TextSize::of(line_docs) - TextSize::of("\n"),
                |(offset, _)| TextSize::new(offset as u32),
            );
            unsafe { guard.0.docs.as_bytes_mut() }.copy_within(
                Range::<usize>::from(TextRange::new(
                    line_source.string_offset + indent_size,
                    string_end_offset,
                )),
                copy_into.into(),
            );
            copy_into += line_docs_len - indent_size;

            if let Some(inner_attrs_start) = &mut guard.0.inline_inner_docs_start
                && *inner_attrs_start == line_source.string_offset
            {
                *inner_attrs_start -= accumulated_offset;
            }
            // The removals in the string accumulate, but in the AST not, because it already points
            // to the beginning of each attribute.
            // Also, we need to shift the AST offset of every line, but the string offset of the first
            // line should not get shifted (in general, the shift for the string offset is by the
            // number of lines until the current one, excluding the current one).
            line_source.string_offset -= accumulated_offset;
            if let Some(ref mut ast_offset) = line_source.ast_offset {
                *ast_offset += indent_size;
            }

            accumulated_offset += indent_size;
        }
        // Don't use `String::truncate()` here because it's not guaranteed to not do UTF-8-dependent things,
        // and we may have temporarily broken the string's encoding.
        unsafe { guard.0.docs.as_mut_vec() }.truncate(copy_into.into());

        std::mem::forget(guard);
    }

    fn remove_last_newline(&mut self) {
        self.docs.truncate(self.docs.len().saturating_sub(1));
    }

    fn shrink_to_fit(&mut self) {
        let Docs {
            docs,
            docs_source_map,
            outline_mod: _,
            inline_file: _,
            prefix_len: _,
            inline_inner_docs_start: _,
            outline_inner_docs_start: _,
        } = self;
        docs.shrink_to_fit();
        docs_source_map.shrink_to_fit();
    }
}

struct DocMacroExpander<'db> {
    db: &'db dyn DefDatabase,
    krate: Crate,
    recursion_depth: usize,
    recursion_limit: usize,
}

struct DocExprSourceCtx<'db> {
    resolver: Resolver<'db>,
    file_id: HirFileId,
    ast_id_map: &'db AstIdMap,
    span_map: SpanMap,
}

fn expand_doc_expr_via_macro_pipeline<'db>(
    expander: &mut DocMacroExpander<'db>,
    source_ctx: &DocExprSourceCtx<'db>,
    expr: ast::Expr,
) -> Option<String> {
    match expr {
        ast::Expr::ParenExpr(paren_expr) => {
            expand_doc_expr_via_macro_pipeline(expander, source_ctx, paren_expr.expr()?)
        }
        ast::Expr::Literal(literal) => match literal.kind() {
            ast::LiteralKind::String(string) => string.value().ok().map(Into::into),
            _ => None,
        },
        ast::Expr::MacroExpr(macro_expr) => {
            let macro_call = macro_expr.macro_call()?;
            let (expr, new_source_ctx) = expand_doc_macro_call(expander, source_ctx, macro_call)?;
            // After expansion, the expr lives in the expansion file; use its source context.
            expand_doc_expr_via_macro_pipeline(expander, &new_source_ctx, expr)
        }
        _ => None,
    }
}

fn expand_doc_macro_call<'db>(
    expander: &mut DocMacroExpander<'db>,
    source_ctx: &DocExprSourceCtx<'db>,
    macro_call: ast::MacroCall,
) -> Option<(ast::Expr, DocExprSourceCtx<'db>)> {
    if expander.recursion_depth >= expander.recursion_limit {
        return None;
    }

    let path = macro_call.path()?;
    let mod_path = ModPath::from_src(expander.db, path, &mut |range| {
        source_ctx.span_map.span_for_range(range).ctx
    })?;
    let call_site = source_ctx.span_map.span_for_range(macro_call.syntax().text_range());
    let ast_id = AstId::new(source_ctx.file_id, source_ctx.ast_id_map.ast_id(&macro_call));
    let call_id = macro_call_as_call_id(
        expander.db,
        ast_id,
        &mod_path,
        call_site.ctx,
        ExpandTo::Expr,
        expander.krate,
        |path| {
            source_ctx.resolver.resolve_path_as_macro_def(expander.db, path, Some(MacroSubNs::Bang))
        },
        &mut |_, _| (),
    )
    .ok()?
    .value?;

    expander.recursion_depth += 1;
    let parse = expander.db.parse_macro_expansion(call_id).value.0;
    let expr = parse.cast::<ast::Expr>().map(|parse| parse.tree())?;
    expander.recursion_depth -= 1;

    // Build a new source context for the expansion file so that any further
    // recursive expansion (e.g. a user macro expanding to `concat!(...)`)
    // correctly resolves AstIds and spans in the expansion.
    let expansion_file_id: HirFileId = call_id.into();
    let new_source_ctx = DocExprSourceCtx {
        resolver: source_ctx.resolver.clone(),
        file_id: expansion_file_id,
        ast_id_map: expander.db.ast_id_map(expansion_file_id),
        span_map: expander.db.span_map(expansion_file_id),
    };
    Some((expr, new_source_ctx))
}

/// Quick check: does this syntax node have any `#[doc = expr]` attributes where the
/// value is not a simple string literal (i.e., it needs macro expansion)?
fn has_doc_macro_attr(node: &SyntaxNode) -> bool {
    ast::AnyHasAttrs::cast(node.clone()).is_some_and(|owner| {
        owner.attrs().any(|attr| {
            let Some(meta) = attr.meta() else { return false };
            // Check it's a `doc` attribute with an expression (e.g. `#[doc = expr]`),
            // but NOT a simple string literal (which wouldn't need macro expansion).
            meta.path().is_some_and(|path| {
                path.as_single_name_ref().is_some_and(|name| name.text() == "doc")
            }) && meta.expr().is_some_and(|expr| !matches!(expr, ast::Expr::Literal(_)))
        })
    })
}

fn extend_with_attrs<'a, 'db>(
    result: &mut Docs,
    node: &SyntaxNode,
    expect_inner_attrs: bool,
    indent: &mut usize,
    get_cfg_options: &dyn Fn() -> &'a CfgOptions,
    cfg_options: &mut Option<&'a CfgOptions>,
    mut expander: Option<&mut DocMacroExpander<'db>>,
    source_ctx: Option<&DocExprSourceCtx<'db>>,
) {
    // FIXME: `#[cfg_attr(..., doc = macro!())]` is not handled correctly here:
    // macro expansion inside `cfg_attr`-wrapped doc attributes is not supported yet.
    // Fixing this properly requires changes to `expand_cfg_attr()`.
    // See https://github.com/rust-lang/rust-analyzer/issues/18444
    expand_cfg_attr_with_doc_comments::<_, Infallible>(
        AttrDocCommentIter::from_syntax_node(node).filter(|attr| match attr {
            Either::Left(attr) => attr.kind().is_inner() == expect_inner_attrs,
            Either::Right(comment) => comment
                .kind()
                .doc
                .is_some_and(|kind| (kind == ast::CommentPlacement::Inner) == expect_inner_attrs),
        }),
        || *cfg_options.get_or_insert_with(get_cfg_options),
        |attr| {
            match attr {
                Either::Right(doc_comment) => result.extend_with_doc_comment(doc_comment, indent),
                Either::Left((attr, _, _, top_attr)) => match attr {
                    Meta::NamedKeyValue { name: Some(name), value: Some(value), .. }
                        if name.text() == "doc" =>
                    {
                        result.extend_with_doc_attr(value, indent);
                    }
                    Meta::NamedKeyValue { name: Some(name), value: None, .. }
                        if name.text() == "doc" =>
                    {
                        if let (Some(expander), Some(source_ctx)) =
                            (expander.as_deref_mut(), source_ctx)
                            && let Some(expr) = top_attr.expr()
                            && let Some(expanded) =
                                expand_doc_expr_via_macro_pipeline(expander, source_ctx, expr)
                        {
                            result.extend_with_unmapped_doc_str(&expanded, indent);
                        }
                    }
                    _ => {}
                },
            }
            ControlFlow::Continue(())
        },
    );
}

pub(crate) fn extract_docs<'a, 'db>(
    db: &'db dyn DefDatabase,
    krate: Crate,
    // For outer docs on an outlined module, use the parent module's resolver.
    // For inline docs (and non-module items), use the item's own resolver.
    outer_resolver: Option<impl FnOnce() -> Resolver<'db>>,
    inline_resolver: impl FnOnce() -> Resolver<'db>,
    get_cfg_options: &dyn Fn() -> &'a CfgOptions,
    source: InFile<ast::AnyHasAttrs>,
    outer_mod_decl: Option<InFile<ast::Module>>,
    inner_attrs_node: Option<SyntaxNode>,
) -> Option<Box<Docs>> {
    let mut result = Docs {
        docs: String::new(),
        docs_source_map: Vec::new(),
        outline_mod: None,
        inline_file: source.file_id,
        prefix_len: TextSize::new(0),
        inline_inner_docs_start: None,
        outline_inner_docs_start: None,
    };

    let mut cfg_options = None;

    if let Some(outer_mod_decl) = outer_mod_decl {
        let mut indent = usize::MAX;
        // For outer docs (the `mod foo;` declaration), use the parent module's resolver
        // so that macros are resolved in the parent's scope.
        let (mut outer_expander, outer_source_ctx) =
            if has_doc_macro_attr(outer_mod_decl.value.syntax())
                && let Some(make) = outer_resolver
            {
                let resolver = make();
                let def_map = resolver.top_level_def_map();
                let recursion_limit = def_map.recursion_limit() as usize;
                let expander = DocMacroExpander { db, krate, recursion_depth: 0, recursion_limit };
                let source_ctx = DocExprSourceCtx {
                    resolver,
                    file_id: outer_mod_decl.file_id,
                    ast_id_map: db.ast_id_map(outer_mod_decl.file_id),
                    span_map: db.span_map(outer_mod_decl.file_id),
                };
                (Some(expander), Some(source_ctx))
            } else {
                (None, None)
            };
        extend_with_attrs(
            &mut result,
            outer_mod_decl.value.syntax(),
            false,
            &mut indent,
            get_cfg_options,
            &mut cfg_options,
            outer_expander.as_mut(),
            outer_source_ctx.as_ref(),
        );
        result.remove_indent(indent, 0);
        result.outline_mod = Some((outer_mod_decl.file_id, result.docs_source_map.len()));
    }

    let inline_source_map_start = result.docs_source_map.len();
    let mut indent = usize::MAX;
    // For inline docs, use the item's own resolver.
    let needs_expansion = has_doc_macro_attr(source.value.syntax())
        || inner_attrs_node.as_ref().is_some_and(has_doc_macro_attr);
    let (mut inline_expander, inline_source_ctx) = if needs_expansion {
        let resolver = inline_resolver();
        let def_map = resolver.top_level_def_map();
        let recursion_limit = def_map.recursion_limit() as usize;
        let expander = DocMacroExpander { db, krate, recursion_depth: 0, recursion_limit };
        let source_ctx = DocExprSourceCtx {
            resolver,
            file_id: source.file_id,
            ast_id_map: db.ast_id_map(source.file_id),
            span_map: db.span_map(source.file_id),
        };
        (Some(expander), Some(source_ctx))
    } else {
        (None, None)
    };
    extend_with_attrs(
        &mut result,
        source.value.syntax(),
        false,
        &mut indent,
        get_cfg_options,
        &mut cfg_options,
        inline_expander.as_mut(),
        inline_source_ctx.as_ref(),
    );
    if let Some(inner_attrs_node) = &inner_attrs_node {
        result.inline_inner_docs_start = Some(TextSize::of(&result.docs));
        extend_with_attrs(
            &mut result,
            inner_attrs_node,
            true,
            &mut indent,
            get_cfg_options,
            &mut cfg_options,
            inline_expander.as_mut(),
            inline_source_ctx.as_ref(),
        );
    }
    result.remove_indent(indent, inline_source_map_start);

    result.remove_last_newline();

    result.shrink_to_fit();

    if result.docs.is_empty() { None } else { Some(Box::new(result)) }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use hir_expand::InFile;
    use test_fixture::WithFixture;
    use tt::{TextRange, TextSize};

    use crate::test_db::TestDB;

    use super::{Docs, IsInnerDoc};

    #[test]
    fn docs() {
        let (_db, file_id) = TestDB::with_single_file("");
        let mut docs = Docs {
            docs: String::new(),
            docs_source_map: Vec::new(),
            outline_mod: None,
            inline_file: file_id.into(),
            prefix_len: TextSize::new(0),
            inline_inner_docs_start: None,
            outline_inner_docs_start: None,
        };
        let mut indent = usize::MAX;

        let outer = " foo\n\tbar  baz";
        let mut ast_offset = TextSize::new(123);
        for line in outer.split('\n') {
            docs.extend_with_doc_str(line, ast_offset, &mut indent);
            ast_offset += TextSize::of(line) + TextSize::of("\n");
        }

        docs.inline_inner_docs_start = Some(TextSize::of(&docs.docs));
        ast_offset += TextSize::new(123);
        let inner = " bar \n baz";
        for line in inner.split('\n') {
            docs.extend_with_doc_str(line, ast_offset, &mut indent);
            ast_offset += TextSize::of(line) + TextSize::of("\n");
        }

        assert_eq!(indent, 1);
        expect![[r#"
            [
                DocsSourceMapLine {
                    string_offset: 0,
                    ast_offset: Some(
                        123,
                    ),
                },
                DocsSourceMapLine {
                    string_offset: 5,
                    ast_offset: Some(
                        128,
                    ),
                },
                DocsSourceMapLine {
                    string_offset: 15,
                    ast_offset: Some(
                        261,
                    ),
                },
                DocsSourceMapLine {
                    string_offset: 20,
                    ast_offset: Some(
                        267,
                    ),
                },
            ]
        "#]]
        .assert_debug_eq(&docs.docs_source_map);

        docs.remove_indent(indent, 0);

        assert_eq!(docs.inline_inner_docs_start, Some(TextSize::new(13)));

        assert_eq!(docs.docs, "foo\nbar  baz\nbar\nbaz\n");
        expect![[r#"
            [
                DocsSourceMapLine {
                    string_offset: 0,
                    ast_offset: Some(
                        124,
                    ),
                },
                DocsSourceMapLine {
                    string_offset: 4,
                    ast_offset: Some(
                        129,
                    ),
                },
                DocsSourceMapLine {
                    string_offset: 13,
                    ast_offset: Some(
                        262,
                    ),
                },
                DocsSourceMapLine {
                    string_offset: 17,
                    ast_offset: Some(
                        268,
                    ),
                },
            ]
        "#]]
        .assert_debug_eq(&docs.docs_source_map);

        docs.append(&docs.clone());
        docs.prepend_str("prefix---");
        assert_eq!(docs.docs, "prefix---foo\nbar  baz\nbar\nbaz\nfoo\nbar  baz\nbar\nbaz\n");
        expect![[r#"
            [
                DocsSourceMapLine {
                    string_offset: 0,
                    ast_offset: Some(
                        124,
                    ),
                },
                DocsSourceMapLine {
                    string_offset: 4,
                    ast_offset: Some(
                        129,
                    ),
                },
                DocsSourceMapLine {
                    string_offset: 13,
                    ast_offset: Some(
                        262,
                    ),
                },
                DocsSourceMapLine {
                    string_offset: 17,
                    ast_offset: Some(
                        268,
                    ),
                },
                DocsSourceMapLine {
                    string_offset: 21,
                    ast_offset: Some(
                        124,
                    ),
                },
                DocsSourceMapLine {
                    string_offset: 25,
                    ast_offset: Some(
                        129,
                    ),
                },
                DocsSourceMapLine {
                    string_offset: 34,
                    ast_offset: Some(
                        262,
                    ),
                },
                DocsSourceMapLine {
                    string_offset: 38,
                    ast_offset: Some(
                        268,
                    ),
                },
            ]
        "#]]
        .assert_debug_eq(&docs.docs_source_map);

        let range = |start, end| TextRange::new(TextSize::new(start), TextSize::new(end));
        let in_file = |range| InFile::new(file_id.into(), range);
        assert_eq!(docs.find_ast_range(range(0, 2)), None);
        assert_eq!(docs.find_ast_range(range(8, 10)), None);
        assert_eq!(
            docs.find_ast_range(range(9, 10)),
            Some((in_file(range(124, 125)), IsInnerDoc::No))
        );
        assert_eq!(docs.find_ast_range(range(20, 23)), None);
        assert_eq!(
            docs.find_ast_range(range(23, 25)),
            Some((in_file(range(263, 265)), IsInnerDoc::Yes))
        );
    }
}
