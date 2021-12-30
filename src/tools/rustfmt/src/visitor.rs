use std::cell::{Cell, RefCell};
use std::rc::Rc;

use rustc_ast::{ast, token::DelimToken, visit, AstLike};
use rustc_data_structures::sync::Lrc;
use rustc_span::{symbol, BytePos, Pos, Span};

use crate::attr::*;
use crate::comment::{contains_comment, rewrite_comment, CodeCharKind, CommentCodeSlices};
use crate::config::Version;
use crate::config::{BraceStyle, Config};
use crate::coverage::transform_missing_snippet;
use crate::items::{
    format_impl, format_trait, format_trait_alias, is_mod_decl, is_use_item, rewrite_extern_crate,
    rewrite_type_alias, FnBraceStyle, FnSig, ItemVisitorKind, StaticParts, StructParts,
};
use crate::macros::{macro_style, rewrite_macro, rewrite_macro_def, MacroPosition};
use crate::modules::Module;
use crate::parse::session::ParseSess;
use crate::rewrite::{Rewrite, RewriteContext};
use crate::shape::{Indent, Shape};
use crate::skip::{is_skip_attr, SkipContext};
use crate::source_map::{LineRangeUtils, SpanUtils};
use crate::spanned::Spanned;
use crate::stmt::Stmt;
use crate::utils::{
    self, contains_skip, count_newlines, depr_skip_annotation, format_unsafety, inner_attributes,
    last_line_width, mk_sp, ptr_vec_to_ref_vec, rewrite_ident, starts_with_newline, stmt_expr,
};
use crate::{ErrorKind, FormatReport, FormattingError};

/// Creates a string slice corresponding to the specified span.
pub(crate) struct SnippetProvider {
    /// A pointer to the content of the file we are formatting.
    big_snippet: Lrc<String>,
    /// A position of the start of `big_snippet`, used as an offset.
    start_pos: usize,
    /// An end position of the file that this snippet lives.
    end_pos: usize,
}

impl SnippetProvider {
    pub(crate) fn span_to_snippet(&self, span: Span) -> Option<&str> {
        let start_index = span.lo().to_usize().checked_sub(self.start_pos)?;
        let end_index = span.hi().to_usize().checked_sub(self.start_pos)?;
        Some(&self.big_snippet[start_index..end_index])
    }

    pub(crate) fn new(start_pos: BytePos, end_pos: BytePos, big_snippet: Lrc<String>) -> Self {
        let start_pos = start_pos.to_usize();
        let end_pos = end_pos.to_usize();
        SnippetProvider {
            big_snippet,
            start_pos,
            end_pos,
        }
    }

    pub(crate) fn entire_snippet(&self) -> &str {
        self.big_snippet.as_str()
    }

    pub(crate) fn start_pos(&self) -> BytePos {
        BytePos::from_usize(self.start_pos)
    }

    pub(crate) fn end_pos(&self) -> BytePos {
        BytePos::from_usize(self.end_pos)
    }
}

pub(crate) struct FmtVisitor<'a> {
    parent_context: Option<&'a RewriteContext<'a>>,
    pub(crate) parse_sess: &'a ParseSess,
    pub(crate) buffer: String,
    pub(crate) last_pos: BytePos,
    // FIXME: use an RAII util or closure for indenting
    pub(crate) block_indent: Indent,
    pub(crate) config: &'a Config,
    pub(crate) is_if_else_block: bool,
    pub(crate) snippet_provider: &'a SnippetProvider,
    pub(crate) line_number: usize,
    /// List of 1-based line ranges which were annotated with skip
    /// Both bounds are inclusifs.
    pub(crate) skipped_range: Rc<RefCell<Vec<(usize, usize)>>>,
    pub(crate) macro_rewrite_failure: bool,
    pub(crate) report: FormatReport,
    pub(crate) skip_context: SkipContext,
    pub(crate) is_macro_def: bool,
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

    fn visit_stmt(&mut self, stmt: &Stmt<'_>, include_empty_semi: bool) {
        debug!(
            "visit_stmt: {}",
            self.parse_sess.span_to_debug_info(stmt.span())
        );

        if stmt.is_empty() {
            // If the statement is empty, just skip over it. Before that, make sure any comment
            // snippet preceding the semicolon is picked up.
            let snippet = self.snippet(mk_sp(self.last_pos, stmt.span().lo()));
            let original_starts_with_newline = snippet
                .find(|c| c != ' ')
                .map_or(false, |i| starts_with_newline(&snippet[i..]));
            let snippet = snippet.trim();
            if !snippet.is_empty() {
                // FIXME(calebcartwright 2021-01-03) - This exists strictly to maintain legacy
                // formatting where rustfmt would preserve redundant semicolons on Items in a
                // statement position.
                // See comment within `walk_stmts` for more info
                if include_empty_semi {
                    self.format_missing(stmt.span().hi());
                } else {
                    if original_starts_with_newline {
                        self.push_str("\n");
                    }

                    self.push_str(&self.block_indent.to_string(self.config));
                    self.push_str(snippet);
                }
            } else if include_empty_semi {
                self.push_str(";");
            }
            self.last_pos = stmt.span().hi();
            return;
        }

        match stmt.as_ast_node().kind {
            ast::StmtKind::Item(ref item) => {
                self.visit_item(item);
                self.last_pos = stmt.span().hi();
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
                    let rewrite = self.with_context(|ctx| stmt.rewrite(ctx, shape));
                    self.push_rewrite(stmt.span(), rewrite)
                }
            }
            ast::StmtKind::MacCall(ref mac_stmt) => {
                if self.visit_attrs(&mac_stmt.attrs, ast::AttrStyle::Outer) {
                    self.push_skipped_with_span(
                        &mac_stmt.attrs,
                        stmt.span(),
                        get_span_without_attrs(stmt.as_ast_node()),
                    );
                } else {
                    self.visit_mac(&mac_stmt.mac, None, MacroPosition::Statement);
                }
                self.format_missing(stmt.span().hi());
            }
            ast::StmtKind::Empty => (),
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
                .next()
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
            "visit_block: {}",
            self.parse_sess.span_to_debug_info(b.span),
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

        let comment_snippet = self.snippet(span);

        let align_to_right = if unindent_comment && contains_comment(comment_snippet) {
            let first_lines = comment_snippet.splitn(2, '/').next().unwrap_or("");
            last_line_width(first_lines) > last_line_width(comment_snippet)
        } else {
            false
        };

        for (kind, offset, sub_slice) in CommentCodeSlices::new(comment_snippet) {
            let sub_slice = transform_missing_snippet(config, sub_slice);

            debug!("close_block: {:?} {:?} {:?}", kind, offset, sub_slice);

            match kind {
                CodeCharKind::Comment => {
                    if !unindented && unindent_comment && !align_to_right {
                        unindented = true;
                        self.block_indent = self.block_indent.block_unindent(config);
                    }
                    let span_in_between = mk_sp(last_hi, span.lo() + BytePos::from_usize(offset));
                    let snippet_in_between = self.snippet(span_in_between);
                    let mut comment_on_same_line = !snippet_in_between.contains('\n');

                    let mut comment_shape =
                        Shape::indented(self.block_indent, config).comment(config);
                    if self.config.version() == Version::Two && comment_on_same_line {
                        self.push_str(" ");
                        // put the first line of the comment on the same line as the
                        // block's last line
                        match sub_slice.find('\n') {
                            None => {
                                self.push_str(&sub_slice);
                            }
                            Some(offset) if offset + 1 == sub_slice.len() => {
                                self.push_str(&sub_slice[..offset]);
                            }
                            Some(offset) => {
                                let first_line = &sub_slice[..offset];
                                self.push_str(first_line);
                                self.push_str(&self.block_indent.to_string_with_newline(config));

                                // put the other lines below it, shaping it as needed
                                let other_lines = &sub_slice[offset + 1..];
                                let comment_str =
                                    rewrite_comment(other_lines, false, comment_shape, config);
                                match comment_str {
                                    Some(ref s) => self.push_str(s),
                                    None => self.push_str(other_lines),
                                }
                            }
                        }
                    } else {
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
    pub(crate) fn visit_fn(
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
            visit::FnKind::Fn(_, ident, _, _, Some(ref b)) => {
                block = b;
                self.rewrite_fn_before_block(
                    indent,
                    ident,
                    &FnSig::from_fn_kind(&fk, generics, fd, defaultness),
                    mk_sp(s.lo(), b.span.lo()),
                )
            }
            _ => unreachable!(),
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
        self.skip_context.update_with_attrs(attrs);

        let should_visit_node_again = match item.kind {
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

        // TODO(calebcartwright): consider enabling box_patterns feature gate
        if should_visit_node_again {
            match item.kind {
                ast::ItemKind::Use(ref tree) => self.format_import(item, tree),
                ast::ItemKind::Impl(ref iimpl) => {
                    let block_indent = self.block_indent;
                    let rw = self.with_context(|ctx| format_impl(ctx, item, iimpl, block_indent));
                    self.push_rewrite(item.span, rw);
                }
                ast::ItemKind::Trait(..) => {
                    let block_indent = self.block_indent;
                    let rw = self.with_context(|ctx| format_trait(ctx, item, block_indent));
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
                ast::ItemKind::Mod(unsafety, ref mod_kind) => {
                    self.format_missing_with_indent(source!(self, item.span).lo());
                    self.format_mod(mod_kind, unsafety, &item.vis, item.span, item.ident, attrs);
                }
                ast::ItemKind::MacCall(ref mac) => {
                    self.visit_mac(mac, Some(item.ident), MacroPosition::Item);
                }
                ast::ItemKind::ForeignMod(ref foreign_mod) => {
                    self.format_missing_with_indent(source!(self, item.span).lo());
                    self.format_foreign_mod(foreign_mod, item.span);
                }
                ast::ItemKind::Static(..) | ast::ItemKind::Const(..) => {
                    self.visit_static(&StaticParts::from_item(item));
                }
                ast::ItemKind::Fn(ref fn_kind) => {
                    let ast::Fn {
                        defaultness,
                        ref sig,
                        ref generics,
                        ref body,
                    } = **fn_kind;
                    if let Some(ref body) = body {
                        let inner_attrs = inner_attributes(&item.attrs);
                        let fn_ctxt = match sig.header.ext {
                            ast::Extern::None => visit::FnCtxt::Free,
                            _ => visit::FnCtxt::Foreign,
                        };
                        self.visit_fn(
                            visit::FnKind::Fn(fn_ctxt, item.ident, sig, &item.vis, Some(body)),
                            generics,
                            &sig.decl,
                            item.span,
                            defaultness,
                            Some(&inner_attrs),
                        )
                    } else {
                        let indent = self.block_indent;
                        let rewrite = self.rewrite_required_fn(
                            indent, item.ident, sig, &item.vis, generics, item.span,
                        );
                        self.push_rewrite(item.span, rewrite);
                    }
                }
                ast::ItemKind::TyAlias(ref ty_alias) => {
                    use ItemVisitorKind::Item;
                    self.visit_ty_alias_kind(ty_alias, &Item(item), item.span);
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

    fn visit_ty_alias_kind(
        &mut self,
        ty_kind: &ast::TyAlias,
        visitor_kind: &ItemVisitorKind<'_>,
        span: Span,
    ) {
        let rewrite = rewrite_type_alias(
            ty_kind,
            &self.get_context(),
            self.block_indent,
            visitor_kind,
            span,
        );
        self.push_rewrite(span, rewrite);
    }

    fn visit_assoc_item(&mut self, visitor_kind: &ItemVisitorKind<'_>) {
        use ItemVisitorKind::*;
        // TODO(calebcartwright): Not sure the skip spans are correct
        let (ai, skip_span, assoc_ctxt) = match visitor_kind {
            AssocTraitItem(ai) => (*ai, ai.span(), visit::AssocCtxt::Trait),
            AssocImplItem(ai) => (*ai, ai.span, visit::AssocCtxt::Impl),
            _ => unreachable!(),
        };
        skip_out_of_file_lines_range_visitor!(self, ai.span);

        if self.visit_attrs(&ai.attrs, ast::AttrStyle::Outer) {
            self.push_skipped_with_span(ai.attrs.as_slice(), skip_span, skip_span);
            return;
        }

        // TODO(calebcartwright): consider enabling box_patterns feature gate
        match (&ai.kind, visitor_kind) {
            (ast::AssocItemKind::Const(..), AssocTraitItem(_)) => {
                self.visit_static(&StaticParts::from_trait_item(ai))
            }
            (ast::AssocItemKind::Const(..), AssocImplItem(_)) => {
                self.visit_static(&StaticParts::from_impl_item(ai))
            }
            (ast::AssocItemKind::Fn(ref fn_kind), _) => {
                let ast::Fn {
                    defaultness,
                    ref sig,
                    ref generics,
                    ref body,
                } = **fn_kind;
                if let Some(ref body) = body {
                    let inner_attrs = inner_attributes(&ai.attrs);
                    let fn_ctxt = visit::FnCtxt::Assoc(assoc_ctxt);
                    self.visit_fn(
                        visit::FnKind::Fn(fn_ctxt, ai.ident, sig, &ai.vis, Some(body)),
                        generics,
                        &sig.decl,
                        ai.span,
                        defaultness,
                        Some(&inner_attrs),
                    );
                } else {
                    let indent = self.block_indent;
                    let rewrite =
                        self.rewrite_required_fn(indent, ai.ident, sig, &ai.vis, generics, ai.span);
                    self.push_rewrite(ai.span, rewrite);
                }
            }
            (ast::AssocItemKind::TyAlias(ref ty_alias), _) => {
                self.visit_ty_alias_kind(ty_alias, visitor_kind, ai.span);
            }
            (ast::AssocItemKind::MacCall(ref mac), _) => {
                self.visit_mac(mac, Some(ai.ident), MacroPosition::Item);
            }
            _ => unreachable!(),
        }
    }

    pub(crate) fn visit_trait_item(&mut self, ti: &ast::AssocItem) {
        self.visit_assoc_item(&ItemVisitorKind::AssocTraitItem(ti));
    }

    pub(crate) fn visit_impl_item(&mut self, ii: &ast::AssocItem) {
        self.visit_assoc_item(&ItemVisitorKind::AssocImplItem(ii));
    }

    fn visit_mac(&mut self, mac: &ast::MacCall, ident: Option<symbol::Ident>, pos: MacroPosition) {
        skip_out_of_file_lines_range_visitor!(self, mac.span());

        // 1 = ;
        let shape = self.shape().saturating_sub_width(1);
        let rewrite = self.with_context(|ctx| rewrite_macro(mac, ident, ctx, shape, pos));
        // As of v638 of the rustc-ap-* crates, the associated span no longer includes
        // the trailing semicolon. This determines the correct span to ensure scenarios
        // with whitespace between the delimiters and trailing semi (i.e. `foo!(abc)     ;`)
        // are formatted correctly.
        let (span, rewrite) = match macro_style(mac, &self.get_context()) {
            DelimToken::Bracket | DelimToken::Paren if MacroPosition::Item == pos => {
                let search_span = mk_sp(mac.span().hi(), self.snippet_provider.end_pos());
                let hi = self.snippet_provider.span_before(search_span, ";");
                let target_span = mk_sp(mac.span().lo(), hi + BytePos(1));
                let rewrite = rewrite.map(|rw| {
                    if !rw.ends_with(';') {
                        format!("{};", rw)
                    } else {
                        rw
                    }
                });
                (target_span, rewrite)
            }
            _ => (mac.span(), rewrite),
        };

        self.push_rewrite(span, rewrite);
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
            self.push_str(snippet.trim());
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
            .map(|attr| self.parse_sess.line_of_byte_pos(attr.span.hi()))
            .max()
            .unwrap_or(1);
        let first_line = self.parse_sess.line_of_byte_pos(main_span.lo());
        // Statement can start after some newlines and/or spaces
        // or it can be on the same line as the last attribute.
        // So here we need to take a minimum between the two.
        let lo = std::cmp::min(attrs_end + 1, first_line);
        self.push_rewrite_inner(item_span, None);
        let hi = self.line_number + 1;
        self.skipped_range.borrow_mut().push((lo, hi));
    }

    pub(crate) fn from_context(ctx: &'a RewriteContext<'_>) -> FmtVisitor<'a> {
        let mut visitor = FmtVisitor::from_parse_sess(
            ctx.parse_sess,
            ctx.config,
            ctx.snippet_provider,
            ctx.report.clone(),
        );
        visitor.skip_context.update(ctx.skip_context.clone());
        visitor.set_parent_context(ctx);
        visitor
    }

    pub(crate) fn from_parse_sess(
        parse_session: &'a ParseSess,
        config: &'a Config,
        snippet_provider: &'a SnippetProvider,
        report: FormatReport,
    ) -> FmtVisitor<'a> {
        FmtVisitor {
            parent_context: None,
            parse_sess: parse_session,
            buffer: String::with_capacity(snippet_provider.big_snippet.len() * 2),
            last_pos: BytePos(0),
            block_indent: Indent::empty(),
            config,
            is_if_else_block: false,
            snippet_provider,
            line_number: 0,
            skipped_range: Rc::new(RefCell::new(vec![])),
            is_macro_def: false,
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
            if attr.has_name(depr_skip_annotation()) {
                let file_name = self.parse_sess.span_to_filename(attr.span);
                self.report.append(
                    file_name,
                    vec![FormattingError::from_span(
                        attr.span,
                        self.parse_sess,
                        ErrorKind::DeprecatedAttr,
                    )],
                );
            } else {
                match &attr.kind {
                    ast::AttrKind::Normal(ref attribute_item, _)
                        if self.is_unknown_rustfmt_attr(&attribute_item.path.segments) =>
                    {
                        let file_name = self.parse_sess.span_to_filename(attr.span);
                        self.report.append(
                            file_name,
                            vec![FormattingError::from_span(
                                attr.span,
                                self.parse_sess,
                                ErrorKind::BadAttr,
                            )],
                        );
                    }
                    _ => (),
                }
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

    fn walk_mod_items(&mut self, items: &[rustc_ast::ptr::P<ast::Item>]) {
        self.visit_items_with_reordering(&ptr_vec_to_ref_vec(items));
    }

    fn walk_stmts(&mut self, stmts: &[Stmt<'_>], include_current_empty_semi: bool) {
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
            self.visit_stmt(&stmts[0], include_current_empty_semi);

            // FIXME(calebcartwright 2021-01-03) - This exists strictly to maintain legacy
            // formatting where rustfmt would preserve redundant semicolons on Items in a
            // statement position.
            //
            // Starting in rustc-ap-* v692 (~2020-12-01) the rustc parser now parses this as
            // two separate statements (Item and Empty kinds), whereas before it was parsed as
            // a single statement with the statement's span including the redundant semicolon.
            //
            // rustfmt typically tosses unnecessary/redundant semicolons, and eventually we
            // should toss these as well, but doing so at this time would
            // break the Stability Guarantee
            // N.B. This could be updated to utilize the version gates.
            let include_next_empty = if stmts.len() > 1 {
                matches!(
                    (&stmts[0].as_ast_node().kind, &stmts[1].as_ast_node().kind),
                    (ast::StmtKind::Item(_), ast::StmtKind::Empty)
                )
            } else {
                false
            };

            self.walk_stmts(&stmts[1..], include_next_empty);
        } else {
            self.visit_items_with_reordering(&items);
            self.walk_stmts(&stmts[items.len()..], false);
        }
    }

    fn walk_block_stmts(&mut self, b: &ast::Block) {
        self.walk_stmts(&Stmt::from_ast_nodes(b.stmts.iter()), false)
    }

    fn format_mod(
        &mut self,
        mod_kind: &ast::ModKind,
        unsafety: ast::Unsafe,
        vis: &ast::Visibility,
        s: Span,
        ident: symbol::Ident,
        attrs: &[ast::Attribute],
    ) {
        let vis_str = utils::format_visibility(&self.get_context(), vis);
        self.push_str(&*vis_str);
        self.push_str(format_unsafety(unsafety));
        self.push_str("mod ");
        // Calling `to_owned()` to work around borrow checker.
        let ident_str = rewrite_ident(&self.get_context(), ident).to_owned();
        self.push_str(&ident_str);

        if let ast::ModKind::Loaded(ref items, ast::Inline::Yes, inner_span) = mod_kind {
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
                self.snippet(mk_sp(mod_lo, source!(self, inner_span).hi() - BytePos(1)));
            let body_snippet = body_snippet.trim();
            if body_snippet.is_empty() {
                self.push_str("}");
            } else {
                self.last_pos = mod_lo;
                self.block_indent = self.block_indent.block_indent(self.config);
                self.visit_attrs(attrs, ast::AttrStyle::Inner);
                self.walk_mod_items(items);
                let missing_span = self.next_span(inner_span.hi() - BytePos(1));
                self.close_block(missing_span, false);
            }
            self.last_pos = source!(self, inner_span).hi();
        } else {
            self.push_str(";");
            self.last_pos = source!(self, s).hi();
        }
    }

    pub(crate) fn format_separate_mod(&mut self, m: &Module<'_>, end_pos: BytePos) {
        self.block_indent = Indent::empty();
        let skipped = self.visit_attrs(m.attrs(), ast::AttrStyle::Inner);
        assert!(
            !skipped,
            "Skipping module must be handled before reaching this line."
        );
        self.walk_mod_items(&m.items);
        self.format_missing_with_indent(end_pos);
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
        let context = self.get_context();
        let result = f(&context);

        self.macro_rewrite_failure |= context.macro_rewrite_failure.get();
        result
    }

    pub(crate) fn get_context(&self) -> RewriteContext<'_> {
        RewriteContext {
            parse_sess: self.parse_sess,
            config: self.config,
            inside_macro: Rc::new(Cell::new(false)),
            use_block: Cell::new(false),
            is_if_else_block: Cell::new(false),
            force_one_line_chain: Cell::new(false),
            snippet_provider: self.snippet_provider,
            macro_rewrite_failure: Cell::new(false),
            is_macro_def: self.is_macro_def,
            report: self.report.clone(),
            skip_context: self.skip_context.clone(),
            skipped_range: self.skipped_range.clone(),
        }
    }
}
