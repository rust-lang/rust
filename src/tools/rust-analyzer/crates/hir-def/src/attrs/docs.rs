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

use base_db::{Crate, SourceDatabase};
use cfg::CfgOptions;
use either::Either;
use hir_expand::{
    AstId, ExpandTo, HirFileId, InFile, MacroCallId,
    attrs::{AstPathExt, expand_cfg_attr_with_doc_comments},
    mod_path::ModPath,
    span_map::SpanMap,
};
use span::AstIdMap;
use syntax::{
    AstNode, AstToken, SyntaxNode,
    ast::{self, AttrDocCommentIter, IsString},
};
use thin_vec::ThinVec;
use tt::{TextRange, TextSize};

use crate::{macro_call_as_call_id, nameres::MacroSubNs, resolver::Resolver};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct DocsSourceMapLine {
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
    /// If the item is an outlined module (`mod foo;`), `docs_source_map` stores the concatenated
    /// list of the outline and inline docs (outline first). Then, this field contains the [`HirFileId`]
    /// of the outline declaration, and the index in `docs` from which the inline docs
    /// begin.
    outline_mod: Option<(HirFileId, usize)>,
    inline_file: HirFileId,
    /// The size of the prepended prefix, which does not map to real doc comments.
    prefix_len: TextSize,
    /// The offset in `docs` from which the docs are inner attributes/comments.
    inline_inner_docs_start: Option<TextSize>,
    /// Like `inline_inner_docs_start`, but for `outline_mod`. This can happen only when merging `Docs`
    /// (as outline modules don't have inner attributes).
    outline_inner_docs_start: Option<TextSize>,
    /// All macro calls in `#[doc = ...]` attributes, recursively.
    macro_calls: ThinVec<(AstId<ast::MacroCall>, MacroCallId)>,
}

#[derive(Clone, Copy)]
enum DocCommentKind {
    /// `///` etc..
    Sugared(ast::CommentShape),
    /// `#[doc = ""]`.
    Desugared,
}

#[derive(Default)]
struct Indent {
    lines: Vec<Option<(usize, DocCommentKind)>>,
    seen_sugared: bool,
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

    #[inline]
    pub fn macro_calls(&self) -> impl Iterator<Item = (AstId<ast::MacroCall>, MacroCallId)> {
        self.macro_calls.iter().copied()
    }

    fn is_empty(&self) -> bool {
        let Self {
            docs,
            docs_source_map: _,
            outline_mod: _,
            inline_file: _,
            prefix_len: _,
            inline_inner_docs_start: _,
            outline_inner_docs_start: _,
            macro_calls,
        } = self;
        docs.is_empty() && macro_calls.is_empty()
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

    fn extend_with_doc_comment(&mut self, comment: ast::Comment, indent: &mut Indent) {
        let Some((doc, offset)) = comment.doc_comment() else { return };
        let offset = comment.syntax().text_range().start() + offset;
        self.extend_with_doc_str(
            doc,
            offset,
            DocCommentKind::Sugared(comment.kind().shape),
            indent,
        );
    }

    fn extend_with_doc_attr(&mut self, value: ast::String, indent: &mut Indent) {
        let Some(value_offset) = value.text_range_between_quotes() else { return };
        let value_offset = value_offset.start();
        let Ok(value) = value.value() else { return };
        // FIXME: Handle source maps for escaped text.
        self.extend_with_doc_str(&value, value_offset, DocCommentKind::Desugared, indent);
    }

    fn extend_with_doc_str(
        &mut self,
        doc: &str,
        offset_in_ast: TextSize,
        comment_kind: DocCommentKind,
        indent: &mut Indent,
    ) {
        self.push_doc_lines(doc, Some(offset_in_ast), comment_kind, indent);
    }

    fn extend_with_unmapped_doc_str(&mut self, doc: &str, indent: &mut Indent) {
        self.push_doc_lines(doc, None, DocCommentKind::Desugared, indent);
    }

    /// Beautifies `doc` and appends the result to `self.docs`, one line at a time via
    /// [`Docs::push_doc_line`]. Mirrors rustc's [`beautify_doc_string`], delegating to
    /// [`get_vertical_trim`] and [`get_horizontal_trim`] for the multi-line case.
    ///
    /// Individual `///` line comments always reach us as a single-line `doc`, so the
    /// `!doc.contains('\n')` fast path fires and the multi-line logic never runs on them.
    /// Desugared `#[doc = "..."]` strings and macro-expanded docs also route through here
    /// with `shape = CommentShape::Line`, matching rustc.
    ///
    /// Unlike rustc's version, which joins the beautified lines into a new interned `Symbol`,
    /// this port pushes each line individually via [`Docs::push_doc_line`] and pairs it with
    /// its byte offset relative to `doc`'s start so the source-map records accurate per-line
    /// offsets.
    ///
    /// [`beautify_doc_string`]: https://github.com/rust-lang/rust/blob/16a623ad672a92409b5c04beb303583c6cf72a7e/compiler/rustc_ast/src/util/comments.rs#L37
    fn push_doc_lines(
        &mut self,
        doc: &str,
        ast_offset: Option<TextSize>,
        comment_kind: DocCommentKind,
        indent: &mut Indent,
    ) {
        // Note: this is pushed even if there are only empty lines here, because that's what rustdoc does.
        let shape = match comment_kind {
            DocCommentKind::Sugared(shape) => {
                indent.seen_sugared = true;
                shape
            }
            // rustc uses `Line` for desugared comments.
            DocCommentKind::Desugared => ast::CommentShape::Line,
        };

        if !doc.contains('\n') {
            self.push_doc_line(doc, ast_offset, comment_kind, indent);
            return;
        }

        let doc_start = doc.as_ptr() as usize;
        let mut lines: Vec<(&str, TextSize)> = doc
            .lines()
            .map(|line| {
                let offset = TextSize::new((line.as_ptr() as usize - doc_start) as u32);
                (line, offset)
            })
            .collect();

        let raw_lines: Vec<&str> = lines.iter().map(|(l, _)| *l).collect();
        let lines = match get_vertical_trim(&raw_lines) {
            Some((i, j)) => &mut lines[i..j],
            None => &mut lines[..],
        };

        let raw_lines: Vec<&str> = lines.iter().map(|(l, _)| *l).collect();
        if let Some(horizontal) = get_horizontal_trim(&raw_lines, shape) {
            let horizontal_len = TextSize::of(horizontal.as_str());
            // Strip `"[ \t]*\*"` from each line where present, exactly like rustc.
            for (line, line_offset) in lines.iter_mut() {
                if let Some(rest) = line.strip_prefix(horizontal.as_str()) {
                    *line = rest;
                    *line_offset += horizontal_len;
                    if shape == ast::CommentShape::Block
                        && (*line == "*" || line.starts_with("* ") || line.starts_with("**"))
                    {
                        *line = &line[1..];
                        *line_offset += TextSize::of("*");
                    }
                }
            }
        }

        for (line, line_offset) in lines.iter().copied() {
            self.push_doc_line(line, ast_offset.map(|it| it + line_offset), comment_kind, indent);
        }
    }

    /// Appends a single beautified line to `self.docs` and records its source-map row.
    fn push_doc_line(
        &mut self,
        line: &str,
        ast_offset: Option<TextSize>,
        comment_kind: DocCommentKind,
        indent: &mut Indent,
    ) {
        self.docs_source_map
            .push(DocsSourceMapLine { string_offset: TextSize::of(&self.docs), ast_offset });

        let line = line.trim_end();
        let line_indent = if line.chars().any(|ch| !ch.is_whitespace()) {
            // Empty lines are handled because `any()` returns `false` for them.
            let line_indent = line.bytes().take_while(|c| *c == b' ' || *c == b'\t').count();
            Some((line_indent, comment_kind))
        } else {
            None
        };
        indent.lines.push(line_indent);
        self.docs.push_str(line);
        self.docs.push('\n');
    }

    fn remove_indent(&mut self, indent: &Indent) {
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
                    macro_calls: _,
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

        // `add` is used in case the most common sugared doc syntax is used ("/// "). The other
        // fragments kind's lines are never starting with a whitespace unless they are using some
        // markdown formatting requiring it. Therefore, if the doc block have a mix between the two,
        // we need to take into account the fact that the minimum indent minus one (to take this
        // whitespace into account).
        //
        // For example:
        //
        // /// hello!
        // #[doc = "another"]
        //
        // In this case, you want "hello! another" and not "hello!  another".
        let add_indent = if indent.seen_sugared { 1 } else { 0 };

        let Some(min_indent) = indent
            .lines
            .iter()
            .filter_map(|it| *it)
            .map(|(line_indent, line_kind)| {
                line_indent
                    + match line_kind {
                        DocCommentKind::Sugared(_) => 0,
                        DocCommentKind::Desugared => add_indent,
                    }
            })
            .min()
        else {
            return;
        };

        let guard = Guard(self);
        let source_map = guard.0.docs_source_map.as_mut_slice();
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
            let indent_size = if let Some((_, DocCommentKind::Desugared)) = indent.lines[idx]
                && min_indent > 0
            {
                min_indent - add_indent
            } else {
                min_indent
            };
            let indent_size = line_docs.char_indices().nth(indent_size).map_or_else(
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
            macro_calls,
        } = self;
        docs.shrink_to_fit();
        docs_source_map.shrink_to_fit();
        macro_calls.shrink_to_fit();
    }
}

/// Copied verbatim from rustc's [`beautify_doc_string`], modulo `CommentKind`/`CommentShape`
/// renaming.
///
/// [`beautify_doc_string`]: https://github.com/rust-lang/rust/blob/16a623ad672a92409b5c04beb303583c6cf72a7e/compiler/rustc_ast/src/util/comments.rs#L38
fn get_vertical_trim(lines: &[&str]) -> Option<(usize, usize)> {
    let mut i = 0;
    let mut j = lines.len();
    // first line of all-stars should be omitted
    if lines.first().is_some_and(|line| line.chars().all(|c| c == '*')) {
        i += 1;
    }

    // like the first, a last line of all stars should be omitted
    if j > i && !lines[j - 1].is_empty() && lines[j - 1].chars().all(|c| c == '*') {
        j -= 1;
    }

    if i != 0 || j != lines.len() { Some((i, j)) } else { None }
}

/// Copied verbatim from rustc's [`beautify_doc_string`], modulo `CommentKind`/`CommentShape`
/// renaming and returning `String` rather than interning to `Symbol`.
///
/// [`beautify_doc_string`]: https://github.com/rust-lang/rust/blob/16a623ad672a92409b5c04beb303583c6cf72a7e/compiler/rustc_ast/src/util/comments.rs#L54
fn get_horizontal_trim(lines: &[&str], kind: ast::CommentShape) -> Option<String> {
    let mut i = usize::MAX;
    let mut first = true;

    // In case we have doc comments like `/**` or `/*!`, we want to remove stars if they are
    // present. However, we first need to strip the empty lines so they don't get in the middle
    // when we try to compute the "horizontal trim".
    let lines = match kind {
        ast::CommentShape::Block => {
            // Whatever happens, we skip the first line.
            let mut i = lines
                .first()
                .map(|l| if l.trim_start().starts_with('*') { 0 } else { 1 })
                .unwrap_or(0);
            let mut j = lines.len();

            while i < j && lines[i].trim().is_empty() {
                i += 1;
            }
            while j > i && lines[j - 1].trim().is_empty() {
                j -= 1;
            }
            &lines[i..j]
        }
        ast::CommentShape::Line => lines,
    };

    for line in lines {
        for (j, c) in line.chars().enumerate() {
            if j > i || !"* \t".contains(c) {
                return None;
            }
            if c == '*' {
                if first {
                    i = j;
                    first = false;
                } else if i != j {
                    return None;
                }
                break;
            }
        }
        if i >= line.len() {
            return None;
        }
    }
    Some(lines.first()?[..i].to_string())
}

struct DocMacroExpander<'db> {
    db: &'db dyn SourceDatabase,
    krate: Crate,
    macro_depth: u32,
    recursion_limit: u32,
    resolver: Resolver<'db>,
    file_id: HirFileId,
    ast_id_map: &'db AstIdMap,
    span_map: SpanMap<'db>,
}

fn expand_doc_expr_via_macro_pipeline<'db>(
    expander: &mut DocMacroExpander<'db>,
    macro_calls: &mut ThinVec<(AstId<ast::MacroCall>, MacroCallId)>,
    expr: ast::Expr,
) -> Option<String> {
    match expr {
        ast::Expr::ParenExpr(paren_expr) => {
            expand_doc_expr_via_macro_pipeline(expander, macro_calls, paren_expr.expr()?)
        }
        ast::Expr::Literal(literal) => match literal.kind() {
            ast::LiteralKind::String(string) => string.value().ok().map(Into::into),
            _ => None,
        },
        ast::Expr::MacroExpr(macro_expr) => {
            let macro_call = macro_expr.macro_call()?;
            expand_doc_macro_call(expander, macro_calls, macro_call)
        }
        _ => None,
    }
}

fn expand_doc_macro_call<'db>(
    expander: &mut DocMacroExpander<'db>,
    macro_calls: &mut ThinVec<(AstId<ast::MacroCall>, MacroCallId)>,
    macro_call: ast::MacroCall,
) -> Option<String> {
    if expander.macro_depth >= expander.recursion_limit {
        return None;
    }

    let path = macro_call.path()?;
    let mod_path = ModPath::from_src(expander.db, path, &mut |range| {
        expander.span_map.span_for_range(range).ctx
    })?;
    let call_site = expander.span_map.span_for_range(macro_call.syntax().text_range());
    let ast_id = AstId::new(expander.file_id, expander.ast_id_map.ast_id(&macro_call));
    let call_id = macro_call_as_call_id(
        expander.db,
        ast_id,
        &mod_path,
        call_site.ctx,
        ExpandTo::Expr,
        expander.krate,
        expander.macro_depth + 1,
        |path| {
            expander.resolver.resolve_path_as_macro_def(expander.db, path, Some(MacroSubNs::Bang))
        },
        &mut |_, _| (),
    )
    .ok()?
    .value?;
    macro_calls.push((ast_id, call_id));

    let (parse, span_map) = &call_id.parse_macro_expansion(expander.db).value;
    let expr = parse.clone().cast::<ast::Expr>().map(|parse| parse.tree())?;

    // Build a new source context for the expansion file so that any further
    // recursive expansion (e.g. a user macro expanding to `concat!(...)`)
    // correctly resolves AstIds and spans in the expansion.
    let expansion_file_id: HirFileId = call_id.into();
    let old_file_id = std::mem::replace(&mut expander.file_id, expansion_file_id);
    let old_span_map =
        std::mem::replace(&mut expander.span_map, SpanMap::ExpansionSpanMap(span_map));
    let old_ast_id_map =
        std::mem::replace(&mut expander.ast_id_map, expansion_file_id.ast_id_map(expander.db));
    expander.macro_depth += 1;

    let expansion = expand_doc_expr_via_macro_pipeline(expander, macro_calls, expr);

    expander.file_id = old_file_id;
    expander.span_map = old_span_map;
    expander.ast_id_map = old_ast_id_map;
    expander.macro_depth -= 1;

    expansion
}

fn extend_with_attrs<'a, 'db>(
    result: &mut Docs,
    db: &'db dyn SourceDatabase,
    krate: Crate,
    node: &SyntaxNode,
    file_id: HirFileId,
    expect_inner_attrs: bool,
    indent: &mut Indent,
    get_cfg_options: &dyn Fn() -> &'a CfgOptions,
    cfg_options: &mut Option<&'a CfgOptions>,
    make_resolver: &dyn Fn() -> Resolver<'db>,
) {
    // Lazily initialised when we first encounter a `#[doc = macro!()]`.
    let mut expander = None;

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
                Either::Left((attr, _)) => match attr {
                    ast::Meta::KeyValueMeta(attr) if attr.path().is1("doc") => {
                        if let Some(value) = attr.expr() {
                            if let ast::Expr::Literal(value) = &value
                                && let ast::LiteralKind::String(value) = value.kind()
                            {
                                result.extend_with_doc_attr(value, indent);
                            } else {
                                let exp = expander.get_or_insert_with(|| {
                                    let resolver = make_resolver();
                                    let def_map = resolver.top_level_def_map();
                                    let recursion_limit = def_map.recursion_limit();
                                    DocMacroExpander {
                                        db,
                                        krate,
                                        macro_depth: file_id.macro_expansion_depth(db),
                                        recursion_limit,
                                        resolver,
                                        file_id,
                                        ast_id_map: file_id.ast_id_map(db),
                                        span_map: file_id.span_map(db),
                                    }
                                });
                                if let Some(expanded) = expand_doc_expr_via_macro_pipeline(
                                    exp,
                                    &mut result.macro_calls,
                                    value,
                                ) {
                                    result.extend_with_unmapped_doc_str(&expanded, indent);
                                }
                            }
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
    db: &'db dyn SourceDatabase,
    krate: Crate,
    resolver: &dyn Fn() -> Resolver<'db>,
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
        macro_calls: ThinVec::new(),
    };

    let mut cfg_options = None;

    let mut indent = Indent::default();
    if let Some(outer_mod_decl) = outer_mod_decl {
        // For outer docs (the `mod foo;` declaration), use the module's own resolver.
        extend_with_attrs(
            &mut result,
            db,
            krate,
            outer_mod_decl.value.syntax(),
            outer_mod_decl.file_id,
            false,
            &mut indent,
            get_cfg_options,
            &mut cfg_options,
            resolver,
        );
        result.outline_mod = Some((outer_mod_decl.file_id, result.docs_source_map.len()));
    }

    // For inline docs, use the item's own resolver.
    extend_with_attrs(
        &mut result,
        db,
        krate,
        source.value.syntax(),
        source.file_id,
        false,
        &mut indent,
        get_cfg_options,
        &mut cfg_options,
        resolver,
    );
    if let Some(inner_attrs_node) = &inner_attrs_node {
        result.inline_inner_docs_start = Some(TextSize::of(&result.docs));
        extend_with_attrs(
            &mut result,
            db,
            krate,
            inner_attrs_node,
            source.file_id,
            true,
            &mut indent,
            get_cfg_options,
            &mut cfg_options,
            resolver,
        );
    }
    result.remove_indent(&indent);

    result.remove_last_newline();

    result.shrink_to_fit();

    if result.is_empty() { None } else { Some(Box::new(result)) }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use hir_expand::InFile;
    use syntax::{AstToken, ast};
    use test_fixture::WithFixture;
    use thin_vec::ThinVec;
    use tt::{TextRange, TextSize};

    use crate::test_db::TestDB;

    use super::{DocCommentKind, Docs, Indent, IsInnerDoc};

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
            macro_calls: ThinVec::new(),
        };
        let mut indent = Indent::default();

        let outer = " foo\n\tbar  baz";
        let mut ast_offset = TextSize::new(123);
        for line in outer.split('\n') {
            docs.extend_with_doc_str(
                line,
                ast_offset,
                DocCommentKind::Sugared(ast::CommentShape::Line),
                &mut indent,
            );
            ast_offset += TextSize::of(line) + TextSize::of("\n");
        }

        docs.inline_inner_docs_start = Some(TextSize::of(&docs.docs));
        ast_offset += TextSize::new(123);
        let inner = " bar \n baz";
        for line in inner.split('\n') {
            docs.extend_with_doc_str(
                line,
                ast_offset,
                DocCommentKind::Sugared(ast::CommentShape::Line),
                &mut indent,
            );
            ast_offset += TextSize::of(line) + TextSize::of("\n");
        }

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

        docs.remove_indent(&indent);

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

    #[test]
    fn sugared_desugared_mix() {
        let (_db, file_id) = TestDB::with_single_file("");
        let mut docs = Docs {
            docs: String::new(),
            docs_source_map: Vec::new(),
            outline_mod: None,
            inline_file: file_id.into(),
            prefix_len: TextSize::new(0),
            inline_inner_docs_start: None,
            outline_inner_docs_start: None,
            macro_calls: ThinVec::new(),
        };
        let mut indent = Indent::default();

        docs.push_doc_lines(
            " hello!",
            None,
            DocCommentKind::Sugared(ast::CommentShape::Line),
            &mut indent,
        );
        docs.push_doc_lines("another", None, DocCommentKind::Desugared, &mut indent);
        docs.remove_indent(&indent);
        docs.remove_last_newline();

        assert_eq!(docs.docs(), "hello!\nanother");
    }

    /// Extracts the docs of the first comment in `source`, running the same normalization as
    /// [`super::extract_docs`] does for inline docs.
    fn comment_docs(source: &str) -> Docs {
        let (_db, file_id) = TestDB::with_single_file("");
        let comment = syntax::SourceFile::parse(source, span::Edition::CURRENT)
            .syntax_node()
            .descendants_with_tokens()
            .filter_map(|it| it.into_token())
            .find_map(ast::Comment::cast)
            .expect("no comment in the fixture");
        let mut docs = Docs {
            docs: String::new(),
            docs_source_map: Vec::new(),
            outline_mod: None,
            inline_file: file_id.into(),
            prefix_len: TextSize::new(0),
            inline_inner_docs_start: None,
            outline_inner_docs_start: None,
            macro_calls: ThinVec::new(),
        };
        let mut indent = Indent::default();
        docs.extend_with_doc_comment(comment, &mut indent);
        docs.remove_indent(&indent);
        docs.remove_last_newline();
        docs
    }

    #[test]
    fn block_doc_comment_stars() {
        #[track_caller]
        fn check(source: &str, expect: expect_test::Expect) {
            expect.assert_eq(&comment_docs(source).docs);
        }

        // The decoration is stripped, but markdown bullets and `*foo` are content.
        // `*bar` doesn't start with `* ` / `**`, so rustc's beautifier only strips the
        // horizontal `[ \t]*` prefix (here a single space) and leaves the leading `*` in
        // place. That in turn pins the block's minimum indent at 0, so surrounding lines
        // aren't re-indented.
        check(
            "/**\n * foo\n *\n *   * bullet\n *bar\n */",
            expect![[r#"
                 foo

                   * bullet
                *bar
            "#]],
        );
        // Single-line block doc comments are left alone, like rustdoc does.
        check("/** * item */", expect!["* item"]);
        // So are blocks without a consistent star column.
        check(
            "/**\n * foo\n   * bar\n */",
            expect![[r#"
                * foo
                  * bar
            "#]],
        );
    }

    #[test]
    fn block_doc_comment_source_map() {
        let docs = comment_docs("/**\n * foo\n * bar\n */");
        // `.lines()` (matching rustc) doesn't emit a leading empty entry for the newline
        // right after `/**`, so the docs body starts at `foo`, not with a blank line.
        assert_eq!(docs.docs, "foo\nbar\n");

        let range = |start, end| TextRange::new(TextSize::new(start), TextSize::new(end));
        let in_file = |range| InFile::new(docs.inline_file, range);
        let mapped = |start, end| docs.find_ast_range(range(start, end));
        // Both `foo` and `bar` map back past the stripped ` * ` decoration.
        assert_eq!(mapped(0, 3), Some((in_file(range(7, 10)), IsInnerDoc::No)));
        assert_eq!(mapped(4, 7), Some((in_file(range(14, 17)), IsInnerDoc::No)));
    }
}
