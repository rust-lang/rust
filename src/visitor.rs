use std::cell::RefCell;

use syntax::parse::ParseSess;
use syntax::source_map::{self, BytePos, Pos, SourceMap, Span};
use syntax::{ast, visit};

use crate::attr::*;
use crate::comment::{rewrite_comment, CodeCharKind, CommentCodeSlices};
use crate::config::{BraceStyle, Config};
use crate::coverage::transform_missing_snippet;
use crate::items::{
    format_impl, format_trait, format_trait_alias, is_mod_decl, is_use_item,
    rewrite_associated_impl_type, rewrite_associated_type, rewrite_existential_impl_type,
    rewrite_existential_type, rewrite_extern_crate, rewrite_type_alias, FnBraceStyle, FnSig,
    StaticParts, StructParts,
};
use crate::macros::{rewrite_macro, rewrite_macro_def, MacroPosition};
use crate::rewrite::{Rewrite, RewriteContext};
use crate::shape::{Indent, Shape};
use crate::skip::{is_skip_attr, SkipContext};
use crate::source_map::{LineRangeUtils, SpanUtils};
use crate::spanned::Spanned;
use crate::stmt::Stmt;
use crate::utils::{
    self, contains_skip, count_newlines, depr_skip_annotation, inner_attributes, last_line_width,
    mk_sp, ptr_vec_to_ref_vec, rewrite_ident, stmt_expr,
};
use crate::{ErrorKind, FormatReport, FormattingError};

/// Creates a string slice corresponding to the specified span.
pub(crate) struct SnippetProvider<'a> {
    /// A pointer to the content of the file we are formatting.
    big_snippet: &'a str,
    /// A position of the start of `big_snippet`, used as an offset.
    start_pos: usize,
}

impl<'a> SnippetProvider<'a> {
    pub(crate) fn span_to_snippet(&self, span: Span) -> Option<&str> {
        let start_index = span.lo().to_usize().checked_sub(self.start_pos)?;
        let end_index = span.hi().to_usize().checked_sub(self.start_pos)?;
        Some(&self.big_snippet[start_index..end_index])
    }

    pub(crate) fn new(start_pos: BytePos, big_snippet: &'a str) -> Self {
        let start_pos = start_pos.to_usize();
        SnippetProvider {
            big_snippet,
            start_pos,
        }
    }
}

pub(crate) struct FmtVisitor<'a> {
    parent_context: Option<&'a RewriteContext<'a>>,
    pub(crate) parse_session: &'a ParseSess,
    pub(crate) source_map: &'a SourceMap,
    pub(crate) buffer: String,
    pub(crate) last_pos: BytePos,
    // FIXME: use an RAII util or closure for indenting
    pub(crate) block_indent: Indent,
    pub(crate) config: &'a Config,
    pub(crate) is_if_else_block: bool,
    pub(crate) snippet_provider: &'a SnippetProvider<'a>,
    pub(crate) line_number: usize,
    /// List of 1-based line ranges which were annotated with skip
    /// Both bounds are inclusifs.
    pub(crate) skipped_range: Vec<(usize, usize)>,
    pub(crate) macro_rewrite_failure: bool,
    pub(crate) report: FormatReport,
    pub(crate) skip_context: SkipContext,
}

impl<'a> Drop for FmtVisitor<'a> {
    fn drop(&mut self) {
        if let Some(ctx) = self.parent_context {
            if self.macro_rewrite_failure {
                ctx.macro_rewrite_failure.replace(true);
            }
        }
    }
}

impl<'b, 'a: 'b> FmtVisitor<'a> {
    fn set_parent_context(&mut self, context: &'a RewriteContext<'_>) {
        self.parent_context = Some(context);
    }

    pub(crate) fn shape(&self) -> Shape {
        Shape::indented(self.block_indent, self.config)
    }

    fn next_span(&self, hi: BytePos) -> Span {
        mk_sp(self.last_pos, hi)
    }

    fn visit_stmt(&mut self, stmt: &Stmt<'_>) {
        debug!(
            "visit_stmt: {:?} {:?}",
            self.source_map.lookup_char_pos(stmt.span().lo()),
            self.source_map.lookup_char_pos(stmt.span().hi())
        );

        match stmt.as_ast_node().node {
            ast::StmtKind::Item(ref item) => {
                self.visit_item(item);
                // Handle potential `;` after the item.
                self.format_missing(stmt.span().hi());
            }
            ast::StmtKind::Local(..) | ast::StmtKind::Expr(..) | ast::StmtKind::Semi(..) => {
                let attrs = get_attrs_from_stmt(stmt.as_ast_node());
                if contains_skip(attrs) {
                    self.push_skipped_with_span(
                        attrs,
                        stmt.span(),
                        get_span_without_attrs(stmt.as_ast_node()),
                    );
                } else {
                    let shape = self.shape();
                    let rewrite = self.with_context(|ctx| stmt.rewrite(&ctx, shape));
                    self.push_rewrite(stmt.span(), rewrite)
                }
            }
            ast::StmtKind::Mac(ref mac) => {
                let (ref mac, _macro_style, ref attrs) = **mac;
                if self.visit_attrs(attrs, ast::AttrStyle::Outer) {
                    self.push_skipped_with_span(
                        attrs,
                        stmt.span(),
                        get_span_without_attrs(stmt.as_ast_node()),
                    );
                } else {
                    self.visit_mac(mac, None, MacroPosition::Statement);
                }
                self.format_missing(stmt.span().hi());
            }
        }
    }

    /// Remove spaces between the opening brace and the first statement or the inner attribute
    /// of the block.
    fn trim_spaces_after_opening_brace(
        &mut self,
        b: &ast::Block,
        inner_attrs: Option<&[ast::Attribute]>,
    ) {
        if let Some(first_stmt) = b.stmts.first() {
            let hi = inner_attrs
                .and_then(|attrs| inner_attributes(attrs).first().map(|attr| attr.span.lo()))
                .unwrap_or_else(|| first_stmt.span().lo());
            let missing_span = self.next_span(hi);
            let snippet = self.snippet(missing_span);
            let len = CommentCodeSlices::new(snippet)
                .nth(0)
                .and_then(|(kind, _, s)| {
                    if kind == CodeCharKind::Normal {
                        s.rfind('\n')
                    } else {
                        None
                    }
                });
            if let Some(len) = len {
                self.last_pos = self.last_pos + BytePos::from_usize(len);
            }
        }
    }

    pub(crate) fn visit_block(
        &mut self,
        b: &ast::Block,
        inner_attrs: Option<&[ast::Attribute]>,
        has_braces: bool,
    ) {
        debug!(
            "visit_block: {:?} {:?}",
            self.source_map.lookup_char_pos(b.span.lo()),
            self.source_map.lookup_char_pos(b.span.hi())
        );

        // Check if this block has braces.
        let brace_compensation = BytePos(if has_braces { 1 } else { 0 });

        self.last_pos = self.last_pos + brace_compensation;
        self.block_indent = self.block_indent.block_indent(self.config);
        self.push_str("{");
        self.trim_spaces_after_opening_brace(b, inner_attrs);

        // Format inner attributes if available.
        if let Some(attrs) = inner_attrs {
            self.visit_attrs(attrs, ast::AttrStyle::Inner);
        }

        self.walk_block_stmts(b);

        if !b.stmts.is_empty() {
            if let Some(expr) = stmt_expr(&b.stmts[b.stmts.len() - 1]) {
                if utils::semicolon_for_expr(&self.get_context(), expr) {
                    self.push_str(";");
                }
            }
        }

        let rest_span = self.next_span(b.span.hi());
        if out_of_file_lines_range!(self, rest_span) {
            self.push_str(self.snippet(rest_span));
            self.block_indent = self.block_indent.block_unindent(self.config);
        } else {
            // Ignore the closing brace.
            let missing_span = self.next_span(b.span.hi() - brace_compensation);
            self.close_block(missing_span, self.unindent_comment_on_closing_brace(b));
        }
        self.last_pos = source!(self, b.span).hi();
    }

    fn close_block(&mut self, span: Span, unindent_comment: bool) {
        let config = self.config;

        let mut last_hi = span.lo();
        let mut unindented = false;
        let mut prev_ends_with_newline = false;
        let mut extra_newline = false;

        let skip_normal = |s: &str| {
            let trimmed = s.trim();
            trimmed.is_empty() || trimmed.chars().all(|c| c == ';')
        };

        for (kind, offset, sub_slice) in CommentCodeSlices::new(self.snippet(span)) {
            let sub_slice = transform_missing_snippet(config, sub_slice);

            debug!("close_block: {:?} {:?} {:?}", kind, offset, sub_slice);

            match kind {
                CodeCharKind::Comment => {
                    if !unindented && unindent_comment {
                        unindented = true;
                        self.block_indent = self.block_indent.block_unindent(config);
                    }
                    let span_in_between = mk_sp(last_hi, span.lo() + BytePos::from_usize(offset));
                    let snippet_in_between = self.snippet(span_in_between);
                    let mut comment_on_same_line = !snippet_in_between.contains("\n");

                    let mut comment_shape =
                        Shape::indented(self.block_indent, config).comment(config);
                    if comment_on_same_line {
                        // 1 = a space before `//`
                        let offset_len = 1 + last_line_width(&self.buffer)
                            .saturating_sub(self.block_indent.width());
                        match comment_shape
                            .visual_indent(offset_len)
                            .sub_width(offset_len)
                        {
                            Some(shp) => comment_shape = shp,
                            None => comment_on_same_line = false,
                        }
                    };

                    if comment_on_same_line {
                        self.push_str(" ");
                    } else {
                        if count_newlines(snippet_in_between) >= 2 || extra_newline {
                            self.push_str("\n");
                        }
                        self.push_str(&self.block_indent.to_string_with_newline(config));
                    }

                    let comment_str = rewrite_comment(&sub_slice, false, comment_shape, config);
                    match comment_str {
                        Some(ref s) => self.push_str(s),
                        None => self.push_str(&sub_slice),
                    }
                }
                CodeCharKind::Normal if skip_normal(&sub_slice) => {
                    extra_newline = prev_ends_with_newline && sub_slice.contains('\n');
                    continue;
                }
                CodeCharKind::Normal => {
                    self.push_str(&self.block_indent.to_string_with_newline(config));
                    self.push_str(sub_slice.trim());
                }
            }
            prev_ends_with_newline = sub_slice.ends_with('\n');
            extra_newline = false;
            last_hi = span.lo() + BytePos::from_usize(offset + sub_slice.len());
        }
        if unindented {
            self.block_indent = self.block_indent.block_indent(self.config);
        }
        self.block_indent = self.block_indent.block_unindent(self.config);
        self.push_str(&self.block_indent.to_string_with_newline(config));
        self.push_str("}");
    }

    fn unindent_comment_on_closing_brace(&self, b: &ast::Block) -> bool {
        self.is_if_else_block && !b.stmts.is_empty()
    }

    // Note that this only gets called for function definitions. Required methods
    // on traits do not get handled here.
    fn visit_fn(
        &mut self,
        fk: visit::FnKind<'_>,
        generics: &ast::Generics,
        fd: &ast::FnDecl,
        s: Span,
        defaultness: ast::Defaultness,
        inner_attrs: Option<&[ast::Attribute]>,
    ) {
        let indent = self.block_indent;
        let block;
        let rewrite = match fk {
            visit::FnKind::ItemFn(ident, _, _, b) | visit::FnKind::Method(ident, _, _, b) => {
                block = b;
                self.rewrite_fn_before_block(
                    indent,
                    ident,
                    &FnSig::from_fn_kind(&fk, generics, fd, defaultness),
                    mk_sp(s.lo(), b.span.lo()),
                )
            }
            visit::FnKind::Closure(_) => unreachable!(),
        };

        if let Some((fn_str, fn_brace_style)) = rewrite {
            self.format_missing_with_indent(source!(self, s).lo());

            if let Some(rw) = self.single_line_fn(&fn_str, block, inner_attrs) {
                self.push_str(&rw);
                self.last_pos = s.hi();
                return;
            }

            self.push_str(&fn_str);
            match fn_brace_style {
                FnBraceStyle::SameLine => self.push_str(" "),
                FnBraceStyle::NextLine => {
                    self.push_str(&self.block_indent.to_string_with_newline(self.config))
                }
                _ => unreachable!(),
            }
            self.last_pos = source!(self, block.span).lo();
        } else {
            self.format_missing(source!(self, block.span).lo());
        }

        self.visit_block(block, inner_attrs, true)
    }

    pub(crate) fn visit_item(&mut self, item: &ast::Item) {
        skip_out_of_file_lines_range_visitor!(self, item.span);

        // This is where we bail out if there is a skip attribute. This is only
        // complex in the module case. It is complex because the module could be
        // in a separate file and there might be attributes in both files, but
        // the AST lumps them all together.
        let filtered_attrs;
        let mut attrs = &item.attrs;
        let skip_context_saved = self.skip_context.clone();
        self.skip_context.update_with_attrs(&attrs);

        let should_visit_node_again = match item.node {
            // For use/extern crate items, skip rewriting attributes but check for a skip attribute.
            ast::ItemKind::Use(..) | ast::ItemKind::ExternCrate(_) => {
                if contains_skip(attrs) {
                    self.push_skipped_with_span(attrs.as_slice(), item.span(), item.span());
                    false
                } else {
                    true
                }
            }
            // Module is inline, in this case we treat it like any other item.
            _ if !is_mod_decl(item) => {
                if self.visit_attrs(&item.attrs, ast::AttrStyle::Outer) {
                    self.push_skipped_with_span(item.attrs.as_slice(), item.span(), item.span());
                    false
                } else {
                    true
                }
            }
            // Module is not inline, but should be skipped.
            ast::ItemKind::Mod(..) if contains_skip(&item.attrs) => false,
            // Module is not inline and should not be skipped. We want
            // to process only the attributes in the current file.
            ast::ItemKind::Mod(..) => {
                filtered_attrs = filter_inline_attrs(&item.attrs, item.span());
                // Assert because if we should skip it should be caught by
                // the above case.
                assert!(!self.visit_attrs(&filtered_attrs, ast::AttrStyle::Outer));
                attrs = &filtered_attrs;
                true
            }
            _ => {
                if self.visit_attrs(&item.attrs, ast::AttrStyle::Outer) {
                    self.push_skipped_with_span(item.attrs.as_slice(), item.span(), item.span());
                    false
                } else {
                    true
                }
            }
        };

        if should_visit_node_again {
            match item.node {
                ast::ItemKind::Use(ref tree) => self.format_import(item, tree),
                ast::ItemKind::Impl(..) => {
                    let block_indent = self.block_indent;
                    let rw = self.with_context(|ctx| format_impl(&ctx, item, block_indent));
                    self.push_rewrite(item.span, rw);
                }
                ast::ItemKind::Trait(..) => {
                    let block_indent = self.block_indent;
                    let rw = self.with_context(|ctx| format_trait(&ctx, item, block_indent));
                    self.push_rewrite(item.span, rw);
                }
                ast::ItemKind::TraitAlias(ref generics, ref generic_bounds) => {
                    let shape = Shape::indented(self.block_indent, self.config);
                    let rw = format_trait_alias(
                        &self.get_context(),
                        item.ident,
                        &item.vis,
                        generics,
                        generic_bounds,
                        shape,
                    );
                    self.push_rewrite(item.span, rw);
                }
                ast::ItemKind::ExternCrate(_) => {
                    let rw = rewrite_extern_crate(&self.get_context(), item, self.shape());
                    let span = if attrs.is_empty() {
                        item.span
                    } else {
                        mk_sp(attrs[0].span.lo(), item.span.hi())
                    };
                    self.push_rewrite(span, rw);
                }
                ast::ItemKind::Struct(..) | ast::ItemKind::Union(..) => {
                    self.visit_struct(&StructParts::from_item(item));
                }
                ast::ItemKind::Enum(ref def, ref generics) => {
                    self.format_missing_with_indent(source!(self, item.span).lo());
                    self.visit_enum(item.ident, &item.vis, def, generics, item.span);
                    self.last_pos = source!(self, item.span).hi();
                }
                ast::ItemKind::Mod(ref module) => {
                    let is_inline = !is_mod_decl(item);
                    self.format_missing_with_indent(source!(self, item.span).lo());
                    self.format_mod(module, &item.vis, item.span, item.ident, attrs, is_inline);
                }
                ast::ItemKind::Mac(ref mac) => {
                    self.visit_mac(mac, Some(item.ident), MacroPosition::Item);
                }
                ast::ItemKind::ForeignMod(ref foreign_mod) => {
                    self.format_missing_with_indent(source!(self, item.span).lo());
                    self.format_foreign_mod(foreign_mod, item.span);
                }
                ast::ItemKind::Static(..) | ast::ItemKind::Const(..) => {
                    self.visit_static(&StaticParts::from_item(item));
                }
                ast::ItemKind::Fn(ref decl, ref fn_header, ref generics, ref body) => {
                    let inner_attrs = inner_attributes(&item.attrs);
                    self.visit_fn(
                        visit::FnKind::ItemFn(item.ident, fn_header, &item.vis, body),
                        generics,
                        decl,
                        item.span,
                        ast::Defaultness::Final,
                        Some(&inner_attrs),
                    )
                }
                ast::ItemKind::Ty(ref ty, ref generics) => {
                    let rewrite = rewrite_type_alias(
                        &self.get_context(),
                        self.block_indent,
                        item.ident,
                        ty,
                        generics,
                        &item.vis,
                    );
                    self.push_rewrite(item.span, rewrite);
                }
                ast::ItemKind::Existential(ref generic_bounds, ref generics) => {
                    let rewrite = rewrite_existential_type(
                        &self.get_context(),
                        self.block_indent,
                        item.ident,
                        generic_bounds,
                        generics,
                        &item.vis,
                    );
                    self.push_rewrite(item.span, rewrite);
                }
                ast::ItemKind::GlobalAsm(..) => {
                    let snippet = Some(self.snippet(item.span).to_owned());
                    self.push_rewrite(item.span, snippet);
                }
                ast::ItemKind::MacroDef(ref def) => {
                    let rewrite = rewrite_macro_def(
                        &self.get_context(),
                        self.shape(),
                        self.block_indent,
                        def,
                        item.ident,
                        &item.vis,
                        item.span,
                    );
                    self.push_rewrite(item.span, rewrite);
                }
            };
        }
        self.skip_context = skip_context_saved;
    }

    pub(crate) fn visit_trait_item(&mut self, ti: &ast::TraitItem) {
        skip_out_of_file_lines_range_visitor!(self, ti.span);

        if self.visit_attrs(&ti.attrs, ast::AttrStyle::Outer) {
            self.push_skipped_with_span(ti.attrs.as_slice(), ti.span(), ti.span());
            return;
        }

        match ti.node {
            ast::TraitItemKind::Const(..) => self.visit_static(&StaticParts::from_trait_item(ti)),
            ast::TraitItemKind::Method(ref sig, None) => {
                let indent = self.block_indent;
                let rewrite =
                    self.rewrite_required_fn(indent, ti.ident, sig, &ti.generics, ti.span);
                self.push_rewrite(ti.span, rewrite);
            }
            ast::TraitItemKind::Method(ref sig, Some(ref body)) => {
                let inner_attrs = inner_attributes(&ti.attrs);
                self.visit_fn(
                    visit::FnKind::Method(ti.ident, sig, None, body),
                    &ti.generics,
                    &sig.decl,
                    ti.span,
                    ast::Defaultness::Final,
                    Some(&inner_attrs),
                );
            }
            ast::TraitItemKind::Type(ref generic_bounds, ref type_default) => {
                let rewrite = rewrite_associated_type(
                    ti.ident,
                    type_default.as_ref(),
                    &ti.generics,
                    Some(generic_bounds),
                    &self.get_context(),
                    self.block_indent,
                );
                self.push_rewrite(ti.span, rewrite);
            }
            ast::TraitItemKind::Macro(ref mac) => {
                self.visit_mac(mac, Some(ti.ident), MacroPosition::Item);
            }
        }
    }

    pub(crate) fn visit_impl_item(&mut self, ii: &ast::ImplItem) {
        skip_out_of_file_lines_range_visitor!(self, ii.span);

        if self.visit_attrs(&ii.attrs, ast::AttrStyle::Outer) {
            self.push_skipped_with_span(ii.attrs.as_slice(), ii.span(), ii.span());
            return;
        }

        match ii.node {
            ast::ImplItemKind::Method(ref sig, ref body) => {
                let inner_attrs = inner_attributes(&ii.attrs);
                self.visit_fn(
                    visit::FnKind::Method(ii.ident, sig, Some(&ii.vis), body),
                    &ii.generics,
                    &sig.decl,
                    ii.span,
                    ii.defaultness,
                    Some(&inner_attrs),
                );
            }
            ast::ImplItemKind::Const(..) => self.visit_static(&StaticParts::from_impl_item(ii)),
            ast::ImplItemKind::Type(ref ty) => {
                let rewrite = rewrite_associated_impl_type(
                    ii.ident,
                    ii.defaultness,
                    Some(ty),
                    &ii.generics,
                    &self.get_context(),
                    self.block_indent,
                );
                self.push_rewrite(ii.span, rewrite);
            }
            ast::ImplItemKind::Existential(ref generic_bounds) => {
                let rewrite = rewrite_existential_impl_type(
                    &self.get_context(),
                    ii.ident,
                    &ii.generics,
                    generic_bounds,
                    self.block_indent,
                );
                self.push_rewrite(ii.span, rewrite);
            }
            ast::ImplItemKind::Macro(ref mac) => {
                self.visit_mac(mac, Some(ii.ident), MacroPosition::Item);
            }
        }
    }

    fn visit_mac(&mut self, mac: &ast::Mac, ident: Option<ast::Ident>, pos: MacroPosition) {
        skip_out_of_file_lines_range_visitor!(self, mac.span);

        // 1 = ;
        let shape = self.shape().saturating_sub_width(1);
        let rewrite = self.with_context(|ctx| rewrite_macro(mac, ident, ctx, shape, pos));
        self.push_rewrite(mac.span, rewrite);
    }

    pub(crate) fn push_str(&mut self, s: &str) {
        self.line_number += count_newlines(s);
        self.buffer.push_str(s);
    }

    #[allow(clippy::needless_pass_by_value)]
    fn push_rewrite_inner(&mut self, span: Span, rewrite: Option<String>) {
        if let Some(ref s) = rewrite {
            self.push_str(s);
        } else {
            let snippet = self.snippet(span);
            self.push_str(snippet);
        }
        self.last_pos = source!(self, span).hi();
    }

    pub(crate) fn push_rewrite(&mut self, span: Span, rewrite: Option<String>) {
        self.format_missing_with_indent(source!(self, span).lo());
        self.push_rewrite_inner(span, rewrite);
    }

    pub(crate) fn push_skipped_with_span(
        &mut self,
        attrs: &[ast::Attribute],
        item_span: Span,
        main_span: Span,
    ) {
        self.format_missing_with_indent(source!(self, item_span).lo());
        // do not take into account the lines with attributes as part of the skipped range
        let attrs_end = attrs
            .iter()
            .map(|attr| self.source_map.lookup_char_pos(attr.span.hi()).line)
            .max()
            .unwrap_or(1);
        let first_line = self.source_map.lookup_char_pos(main_span.lo()).line;
        // Statement can start after some newlines and/or spaces
        // or it can be on the same line as the last attribute.
        // So here we need to take a minimum between the two.
        let lo = std::cmp::min(attrs_end + 1, first_line);
        self.push_rewrite_inner(item_span, None);
        let hi = self.line_number + 1;
        self.skipped_range.push((lo, hi));
    }

    pub(crate) fn from_context(ctx: &'a RewriteContext<'_>) -> FmtVisitor<'a> {
        let mut visitor = FmtVisitor::from_source_map(
            ctx.parse_session,
            ctx.config,
            ctx.snippet_provider,
            ctx.report.clone(),
        );
        visitor.skip_context.update(ctx.skip_context.clone());
        visitor.set_parent_context(ctx);
        visitor
    }

    pub(crate) fn from_source_map(
        parse_session: &'a ParseSess,
        config: &'a Config,
        snippet_provider: &'a SnippetProvider<'_>,
        report: FormatReport,
    ) -> FmtVisitor<'a> {
        FmtVisitor {
            parent_context: None,
            parse_session,
            source_map: parse_session.source_map(),
            buffer: String::with_capacity(snippet_provider.big_snippet.len() * 2),
            last_pos: BytePos(0),
            block_indent: Indent::empty(),
            config,
            is_if_else_block: false,
            snippet_provider,
            line_number: 0,
            skipped_range: vec![],
            macro_rewrite_failure: false,
            report,
            skip_context: Default::default(),
        }
    }

    pub(crate) fn opt_snippet(&'b self, span: Span) -> Option<&'a str> {
        self.snippet_provider.span_to_snippet(span)
    }

    pub(crate) fn snippet(&'b self, span: Span) -> &'a str {
        self.opt_snippet(span).unwrap()
    }

    // Returns true if we should skip the following item.
    pub(crate) fn visit_attrs(&mut self, attrs: &[ast::Attribute], style: ast::AttrStyle) -> bool {
        for attr in attrs {
            if attr.check_name(depr_skip_annotation()) {
                let file_name = self.source_map.span_to_filename(attr.span).into();
                self.report.append(
                    file_name,
                    vec![FormattingError::from_span(
                        attr.span,
                        &self.source_map,
                        ErrorKind::DeprecatedAttr,
                    )],
                );
            } else if self.is_unknown_rustfmt_attr(&attr.path.segments) {
                let file_name = self.source_map.span_to_filename(attr.span).into();
                self.report.append(
                    file_name,
                    vec![FormattingError::from_span(
                        attr.span,
                        &self.source_map,
                        ErrorKind::BadAttr,
                    )],
                );
            }
        }
        if contains_skip(attrs) {
            return true;
        }

        let attrs: Vec<_> = attrs.iter().filter(|a| a.style == style).cloned().collect();
        if attrs.is_empty() {
            return false;
        }

        let rewrite = attrs.rewrite(&self.get_context(), self.shape());
        let span = mk_sp(attrs[0].span.lo(), attrs[attrs.len() - 1].span.hi());
        self.push_rewrite(span, rewrite);

        false
    }

    fn is_unknown_rustfmt_attr(&self, segments: &[ast::PathSegment]) -> bool {
        if segments[0].ident.to_string() != "rustfmt" {
            return false;
        }
        !is_skip_attr(segments)
    }

    fn walk_mod_items(&mut self, m: &ast::Mod) {
        self.visit_items_with_reordering(&ptr_vec_to_ref_vec(&m.items));
    }

    fn walk_stmts(&mut self, stmts: &[Stmt<'_>]) {
        if stmts.is_empty() {
            return;
        }

        // Extract leading `use ...;`.
        let items: Vec<_> = stmts
            .iter()
            .take_while(|stmt| stmt.to_item().map_or(false, is_use_item))
            .filter_map(|stmt| stmt.to_item())
            .collect();

        if items.is_empty() {
            self.visit_stmt(&stmts[0]);
            self.walk_stmts(&stmts[1..]);
        } else {
            self.visit_items_with_reordering(&items);
            self.walk_stmts(&stmts[items.len()..]);
        }
    }

    fn walk_block_stmts(&mut self, b: &ast::Block) {
        self.walk_stmts(&Stmt::from_ast_nodes(b.stmts.iter()))
    }

    fn format_mod(
        &mut self,
        m: &ast::Mod,
        vis: &ast::Visibility,
        s: Span,
        ident: ast::Ident,
        attrs: &[ast::Attribute],
        is_internal: bool,
    ) {
        let vis_str = utils::format_visibility(&self.get_context(), vis);
        self.push_str(&*vis_str);
        self.push_str("mod ");
        // Calling `to_owned()` to work around borrow checker.
        let ident_str = rewrite_ident(&self.get_context(), ident).to_owned();
        self.push_str(&ident_str);

        if is_internal {
            match self.config.brace_style() {
                BraceStyle::AlwaysNextLine => {
                    let indent_str = self.block_indent.to_string_with_newline(self.config);
                    self.push_str(&indent_str);
                    self.push_str("{");
                }
                _ => self.push_str(" {"),
            }
            // Hackery to account for the closing }.
            let mod_lo = self.snippet_provider.span_after(source!(self, s), "{");
            let body_snippet =
                self.snippet(mk_sp(mod_lo, source!(self, m.inner).hi() - BytePos(1)));
            let body_snippet = body_snippet.trim();
            if body_snippet.is_empty() {
                self.push_str("}");
            } else {
                self.last_pos = mod_lo;
                self.block_indent = self.block_indent.block_indent(self.config);
                self.visit_attrs(attrs, ast::AttrStyle::Inner);
                self.walk_mod_items(m);
                let missing_span = self.next_span(m.inner.hi() - BytePos(1));
                self.close_block(missing_span, false);
            }
            self.last_pos = source!(self, m.inner).hi();
        } else {
            self.push_str(";");
            self.last_pos = source!(self, s).hi();
        }
    }

    pub(crate) fn format_separate_mod(
        &mut self,
        m: &ast::Mod,
        source_file: &source_map::SourceFile,
    ) {
        self.block_indent = Indent::empty();
        self.walk_mod_items(m);
        self.format_missing_with_indent(source_file.end_pos);
    }

    pub(crate) fn skip_empty_lines(&mut self, end_pos: BytePos) {
        while let Some(pos) = self
            .snippet_provider
            .opt_span_after(self.next_span(end_pos), "\n")
        {
            if let Some(snippet) = self.opt_snippet(self.next_span(pos)) {
                if snippet.trim().is_empty() {
                    self.last_pos = pos;
                } else {
                    return;
                }
            }
        }
    }

    pub(crate) fn with_context<F>(&mut self, f: F) -> Option<String>
    where
        F: Fn(&RewriteContext<'_>) -> Option<String>,
    {
        // FIXME borrow checker fighting - can be simplified a lot with NLL.
        let (result, mrf) = {
            let context = self.get_context();
            let result = f(&context);
            let mrf = &context.macro_rewrite_failure.borrow();
            (result, *std::ops::Deref::deref(mrf))
        };

        self.macro_rewrite_failure |= mrf;
        result
    }

    pub(crate) fn get_context(&self) -> RewriteContext<'_> {
        RewriteContext {
            parse_session: self.parse_session,
            source_map: self.source_map,
            config: self.config,
            inside_macro: RefCell::new(false),
            use_block: RefCell::new(false),
            is_if_else_block: RefCell::new(false),
            force_one_line_chain: RefCell::new(false),
            snippet_provider: self.snippet_provider,
            macro_rewrite_failure: RefCell::new(false),
            report: self.report.clone(),
            skip_context: self.skip_context.clone(),
        }
    }
}
