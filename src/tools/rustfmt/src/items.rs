// Formatting top-level items - functions, structs, enums, traits, impls.

use std::borrow::Cow;
use std::cmp::{Ordering, max, min};

use regex::Regex;
use rustc_ast::ast;
use rustc_ast::visit;
use rustc_span::{BytePos, DUMMY_SP, Ident, Span, symbol};
use tracing::debug;

use crate::attr::filter_inline_attrs;
use crate::comment::{
    FindUncommented, combine_strs_with_missing_comments, contains_comment, is_last_comment_block,
    recover_comment_removed, recover_missing_comment_in_span, rewrite_missing_comment,
};
use crate::config::lists::*;
use crate::config::{BraceStyle, Config, IndentStyle, StyleEdition};
use crate::expr::{
    RhsAssignKind, RhsTactics, is_empty_block, is_simple_block_stmt, rewrite_assign_rhs,
    rewrite_assign_rhs_with, rewrite_assign_rhs_with_comments, rewrite_else_kw_with_comments,
    rewrite_let_else_block,
};
use crate::lists::{ListFormatting, Separator, definitive_tactic, itemize_list, write_list};
use crate::macros::{MacroPosition, rewrite_macro};
use crate::overflow;
use crate::rewrite::{Rewrite, RewriteContext, RewriteError, RewriteErrorExt, RewriteResult};
use crate::shape::{Indent, Shape};
use crate::source_map::{LineRangeUtils, SpanUtils};
use crate::spanned::Spanned;
use crate::stmt::Stmt;
use crate::types::opaque_ty;
use crate::utils::*;
use crate::vertical::rewrite_with_alignment;
use crate::visitor::FmtVisitor;

const DEFAULT_VISIBILITY: ast::Visibility = ast::Visibility {
    kind: ast::VisibilityKind::Inherited,
    span: DUMMY_SP,
    tokens: None,
};

fn type_annotation_separator(config: &Config) -> &str {
    colon_spaces(config)
}

// Statements of the form
// let pat: ty = init; or let pat: ty = init else { .. };
impl Rewrite for ast::Local {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        self.rewrite_result(context, shape).ok()
    }

    fn rewrite_result(&self, context: &RewriteContext<'_>, shape: Shape) -> RewriteResult {
        debug!(
            "Local::rewrite {:?} {} {:?}",
            self, shape.width, shape.indent
        );

        skip_out_of_file_lines_range_err!(context, self.span);

        if contains_skip(&self.attrs) {
            return Err(RewriteError::SkipFormatting);
        }

        // FIXME(super_let): Implement formatting
        if self.super_.is_some() {
            return Err(RewriteError::SkipFormatting);
        }

        let attrs_str = self.attrs.rewrite_result(context, shape)?;
        let mut result = if attrs_str.is_empty() {
            "let ".to_owned()
        } else {
            combine_strs_with_missing_comments(
                context,
                &attrs_str,
                "let ",
                mk_sp(
                    self.attrs.last().map(|a| a.span.hi()).unwrap(),
                    self.span.lo(),
                ),
                shape,
                false,
            )?
        };
        let let_kw_offset = result.len() - "let ".len();

        // 4 = "let ".len()
        let pat_shape = shape
            .offset_left(4)
            .max_width_error(shape.width, self.span())?;
        // 1 = ;
        let pat_shape = pat_shape
            .sub_width(1)
            .max_width_error(shape.width, self.span())?;
        let pat_str = self.pat.rewrite_result(context, pat_shape)?;

        result.push_str(&pat_str);

        // String that is placed within the assignment pattern and expression.
        let infix = {
            let mut infix = String::with_capacity(32);

            if let Some(ref ty) = self.ty {
                let separator = type_annotation_separator(context.config);
                let ty_shape = if pat_str.contains('\n') {
                    shape.with_max_width(context.config)
                } else {
                    shape
                }
                .offset_left(last_line_width(&result) + separator.len())
                .max_width_error(shape.width, self.span())?
                // 2 = ` =`
                .sub_width(2)
                .max_width_error(shape.width, self.span())?;

                let rewrite = ty.rewrite_result(context, ty_shape)?;

                infix.push_str(separator);
                infix.push_str(&rewrite);
            }

            if self.kind.init().is_some() {
                infix.push_str(" =");
            }

            infix
        };

        result.push_str(&infix);

        if let Some((init, else_block)) = self.kind.init_else_opt() {
            // 1 = trailing semicolon;
            let nested_shape = shape
                .sub_width(1)
                .max_width_error(shape.width, self.span())?;

            result = rewrite_assign_rhs(
                context,
                result,
                init,
                &RhsAssignKind::Expr(&init.kind, init.span),
                nested_shape,
            )?;

            if let Some(block) = else_block {
                let else_kw_span = init.span.between(block.span);
                // Strip attributes and comments to check if newline is needed before the else
                // keyword from the initializer part. (#5901)
                let style_edition = context.config.style_edition();
                let init_str = if style_edition >= StyleEdition::Edition2024 {
                    &result[let_kw_offset..]
                } else {
                    result.as_str()
                };
                let force_newline_else = pat_str.contains('\n')
                    || !same_line_else_kw_and_brace(init_str, context, else_kw_span, nested_shape);
                let else_kw = rewrite_else_kw_with_comments(
                    force_newline_else,
                    true,
                    context,
                    else_kw_span,
                    shape,
                );
                result.push_str(&else_kw);

                // At this point we've written `let {pat} = {expr} else' into the buffer, and we
                // want to calculate up front if there's room to write the divergent block on the
                // same line. The available space varies based on indentation so we clamp the width
                // on the smaller of `shape.width` and `single_line_let_else_max_width`.
                let max_width =
                    std::cmp::min(shape.width, context.config.single_line_let_else_max_width());

                // If available_space hits zero we know for sure this will be a multi-lined block
                let style_edition = context.config.style_edition();
                let assign_str_with_else_kw = if style_edition >= StyleEdition::Edition2024 {
                    &result[let_kw_offset..]
                } else {
                    result.as_str()
                };
                let available_space = max_width.saturating_sub(assign_str_with_else_kw.len());

                let allow_single_line = !force_newline_else
                    && available_space > 0
                    && allow_single_line_let_else_block(assign_str_with_else_kw, block);

                let mut rw_else_block =
                    rewrite_let_else_block(block, allow_single_line, context, shape)?;

                let single_line_else = !rw_else_block.contains('\n');
                // +1 for the trailing `;`
                let else_block_exceeds_width = rw_else_block.len() + 1 > available_space;

                if allow_single_line && single_line_else && else_block_exceeds_width {
                    // writing this on one line would exceed the available width
                    // so rewrite the else block over multiple lines.
                    rw_else_block = rewrite_let_else_block(block, false, context, shape)?;
                }

                result.push_str(&rw_else_block);
            };
        }

        result.push(';');
        Ok(result)
    }
}

/// When the initializer expression is multi-lined, then the else keyword and opening brace of the
/// block ( i.e. "else {") should be put on the same line as the end of the initializer expression
/// if all the following are true:
///
/// 1. The initializer expression ends with one or more closing parentheses, square brackets,
///    or braces
/// 2. There is nothing else on that line
/// 3. That line is not indented beyond the indent on the first line of the let keyword
fn same_line_else_kw_and_brace(
    init_str: &str,
    context: &RewriteContext<'_>,
    else_kw_span: Span,
    init_shape: Shape,
) -> bool {
    if !init_str.contains('\n') {
        // initializer expression is single lined. The "else {" can only be placed on the same line
        // as the initializer expression if there is enough room for it.
        // 7 = ` else {`
        return init_shape.width.saturating_sub(init_str.len()) >= 7;
    }

    // 1. The initializer expression ends with one or more `)`, `]`, `}`.
    if !init_str.ends_with([')', ']', '}']) {
        return false;
    }

    // 2. There is nothing else on that line
    // For example, there are no comments
    let else_kw_snippet = context.snippet(else_kw_span).trim();
    if else_kw_snippet != "else" {
        return false;
    }

    // 3. The last line of the initializer expression is not indented beyond the `let` keyword
    let indent = init_shape.indent.to_string(context.config);
    init_str
        .lines()
        .last()
        .expect("initializer expression is multi-lined")
        .strip_prefix(indent.as_ref())
        .map_or(false, |l| !l.starts_with(char::is_whitespace))
}

fn allow_single_line_let_else_block(result: &str, block: &ast::Block) -> bool {
    if result.contains('\n') {
        return false;
    }

    if block.stmts.len() <= 1 {
        return true;
    }

    false
}

// FIXME convert to using rewrite style rather than visitor
// FIXME format modules in this style
#[allow(dead_code)]
#[derive(Debug)]
struct Item<'a> {
    safety: ast::Safety,
    abi: Cow<'static, str>,
    vis: Option<&'a ast::Visibility>,
    body: Vec<BodyElement<'a>>,
    span: Span,
}

impl<'a> Item<'a> {
    fn from_foreign_mod(fm: &'a ast::ForeignMod, span: Span, config: &Config) -> Item<'a> {
        Item {
            safety: fm.safety,
            abi: format_extern(
                ast::Extern::from_abi(fm.abi, DUMMY_SP),
                config.force_explicit_abi(),
            ),
            vis: None,
            body: fm
                .items
                .iter()
                .map(|i| BodyElement::ForeignItem(i))
                .collect(),
            span,
        }
    }
}

#[derive(Debug)]
enum BodyElement<'a> {
    // Stmt(&'a ast::Stmt),
    // Field(&'a ast::ExprField),
    // Variant(&'a ast::Variant),
    // Item(&'a ast::Item),
    ForeignItem(&'a ast::ForeignItem),
}

/// Represents a fn's signature.
pub(crate) struct FnSig<'a> {
    decl: &'a ast::FnDecl,
    generics: &'a ast::Generics,
    ext: ast::Extern,
    coroutine_kind: Cow<'a, Option<ast::CoroutineKind>>,
    constness: ast::Const,
    defaultness: ast::Defaultness,
    safety: ast::Safety,
    visibility: &'a ast::Visibility,
}

impl<'a> FnSig<'a> {
    pub(crate) fn from_method_sig(
        method_sig: &'a ast::FnSig,
        generics: &'a ast::Generics,
        visibility: &'a ast::Visibility,
    ) -> FnSig<'a> {
        FnSig {
            safety: method_sig.header.safety,
            coroutine_kind: Cow::Borrowed(&method_sig.header.coroutine_kind),
            constness: method_sig.header.constness,
            defaultness: ast::Defaultness::Final,
            ext: method_sig.header.ext,
            decl: &*method_sig.decl,
            generics,
            visibility,
        }
    }

    pub(crate) fn from_fn_kind(
        fn_kind: &'a visit::FnKind<'_>,
        decl: &'a ast::FnDecl,
        defaultness: ast::Defaultness,
    ) -> FnSig<'a> {
        match *fn_kind {
            visit::FnKind::Fn(visit::FnCtxt::Assoc(..), vis, ast::Fn { sig, generics, .. }) => {
                let mut fn_sig = FnSig::from_method_sig(sig, generics, vis);
                fn_sig.defaultness = defaultness;
                fn_sig
            }
            visit::FnKind::Fn(_, vis, ast::Fn { sig, generics, .. }) => FnSig {
                decl,
                generics,
                ext: sig.header.ext,
                constness: sig.header.constness,
                coroutine_kind: Cow::Borrowed(&sig.header.coroutine_kind),
                defaultness,
                safety: sig.header.safety,
                visibility: vis,
            },
            _ => unreachable!(),
        }
    }

    fn to_str(&self, context: &RewriteContext<'_>) -> String {
        let mut result = String::with_capacity(128);
        // Vis defaultness constness unsafety abi.
        result.push_str(&*format_visibility(context, self.visibility));
        result.push_str(format_defaultness(self.defaultness));
        result.push_str(format_constness(self.constness));
        self.coroutine_kind
            .map(|coroutine_kind| result.push_str(format_coro(&coroutine_kind)));
        result.push_str(format_safety(self.safety));
        result.push_str(&format_extern(
            self.ext,
            context.config.force_explicit_abi(),
        ));
        result
    }
}

impl<'a> FmtVisitor<'a> {
    fn format_item(&mut self, item: &Item<'_>) {
        self.buffer.push_str(format_safety(item.safety));
        self.buffer.push_str(&item.abi);

        let snippet = self.snippet(item.span);
        let brace_pos = snippet.find_uncommented("{").unwrap();

        self.push_str("{");
        if !item.body.is_empty() || contains_comment(&snippet[brace_pos..]) {
            // FIXME: this skips comments between the extern keyword and the opening
            // brace.
            self.last_pos = item.span.lo() + BytePos(brace_pos as u32 + 1);
            self.block_indent = self.block_indent.block_indent(self.config);

            if !item.body.is_empty() {
                for item in &item.body {
                    self.format_body_element(item);
                }
            }

            self.format_missing_no_indent(item.span.hi() - BytePos(1));
            self.block_indent = self.block_indent.block_unindent(self.config);
            let indent_str = self.block_indent.to_string(self.config);
            self.push_str(&indent_str);
        }

        self.push_str("}");
        self.last_pos = item.span.hi();
    }

    fn format_body_element(&mut self, element: &BodyElement<'_>) {
        match *element {
            BodyElement::ForeignItem(item) => self.format_foreign_item(item),
        }
    }

    pub(crate) fn format_foreign_mod(&mut self, fm: &ast::ForeignMod, span: Span) {
        let item = Item::from_foreign_mod(fm, span, self.config);
        self.format_item(&item);
    }

    fn format_foreign_item(&mut self, item: &ast::ForeignItem) {
        let rewrite = item.rewrite(&self.get_context(), self.shape());
        let hi = item.span.hi();
        let span = if item.attrs.is_empty() {
            item.span
        } else {
            mk_sp(item.attrs[0].span.lo(), hi)
        };
        self.push_rewrite(span, rewrite);
        self.last_pos = hi;
    }

    pub(crate) fn rewrite_fn_before_block(
        &mut self,
        indent: Indent,
        ident: symbol::Ident,
        fn_sig: &FnSig<'_>,
        span: Span,
    ) -> Option<(String, FnBraceStyle)> {
        let context = self.get_context();

        let mut fn_brace_style = newline_for_brace(self.config, &fn_sig.generics.where_clause);
        let (result, _, force_newline_brace) =
            rewrite_fn_base(&context, indent, ident, fn_sig, span, fn_brace_style).ok()?;

        // 2 = ` {`
        if self.config.brace_style() == BraceStyle::AlwaysNextLine
            || force_newline_brace
            || last_line_width(&result) + 2 > self.shape().width
        {
            fn_brace_style = FnBraceStyle::NextLine
        }

        Some((result, fn_brace_style))
    }

    pub(crate) fn rewrite_required_fn(
        &mut self,
        indent: Indent,
        ident: symbol::Ident,
        sig: &ast::FnSig,
        vis: &ast::Visibility,
        generics: &ast::Generics,
        span: Span,
    ) -> RewriteResult {
        // Drop semicolon or it will be interpreted as comment.
        let span = mk_sp(span.lo(), span.hi() - BytePos(1));
        let context = self.get_context();

        let (mut result, ends_with_comment, _) = rewrite_fn_base(
            &context,
            indent,
            ident,
            &FnSig::from_method_sig(sig, generics, vis),
            span,
            FnBraceStyle::None,
        )?;

        // If `result` ends with a comment, then remember to add a newline
        if ends_with_comment {
            result.push_str(&indent.to_string_with_newline(context.config));
        }

        // Re-attach semicolon
        result.push(';');

        Ok(result)
    }

    pub(crate) fn single_line_fn(
        &self,
        fn_str: &str,
        block: &ast::Block,
        inner_attrs: Option<&[ast::Attribute]>,
    ) -> Option<String> {
        if fn_str.contains('\n') || inner_attrs.map_or(false, |a| !a.is_empty()) {
            return None;
        }

        let context = self.get_context();

        if self.config.empty_item_single_line()
            && is_empty_block(&context, block, None)
            && self.block_indent.width() + fn_str.len() + 3 <= self.config.max_width()
            && !last_line_contains_single_line_comment(fn_str)
        {
            return Some(format!("{fn_str} {{}}"));
        }

        if !self.config.fn_single_line() || !is_simple_block_stmt(&context, block, None) {
            return None;
        }

        let res = Stmt::from_ast_node(block.stmts.first()?, true)
            .rewrite(&self.get_context(), self.shape())?;

        let width = self.block_indent.width() + fn_str.len() + res.len() + 5;
        if !res.contains('\n') && width <= self.config.max_width() {
            Some(format!("{fn_str} {{ {res} }}"))
        } else {
            None
        }
    }

    pub(crate) fn visit_static(&mut self, static_parts: &StaticParts<'_>) {
        let rewrite = rewrite_static(&self.get_context(), static_parts, self.block_indent);
        self.push_rewrite(static_parts.span, rewrite);
    }

    pub(crate) fn visit_struct(&mut self, struct_parts: &StructParts<'_>) {
        let is_tuple = match struct_parts.def {
            ast::VariantData::Tuple(..) => true,
            _ => false,
        };
        let rewrite = format_struct(&self.get_context(), struct_parts, self.block_indent, None)
            .map(|s| if is_tuple { s + ";" } else { s });
        self.push_rewrite(struct_parts.span, rewrite);
    }

    pub(crate) fn visit_enum(
        &mut self,
        ident: symbol::Ident,
        vis: &ast::Visibility,
        enum_def: &ast::EnumDef,
        generics: &ast::Generics,
        span: Span,
    ) {
        let enum_header =
            format_header(&self.get_context(), "enum ", ident, vis, self.block_indent);
        self.push_str(&enum_header);

        let enum_snippet = self.snippet(span);
        let brace_pos = enum_snippet.find_uncommented("{").unwrap();
        let body_start = span.lo() + BytePos(brace_pos as u32 + 1);
        let generics_str = format_generics(
            &self.get_context(),
            generics,
            self.config.brace_style(),
            if enum_def.variants.is_empty() {
                BracePos::ForceSameLine
            } else {
                BracePos::Auto
            },
            self.block_indent,
            // make a span that starts right after `enum Foo`
            mk_sp(ident.span.hi(), body_start),
            last_line_width(&enum_header),
        )
        .unwrap();
        self.push_str(&generics_str);

        self.last_pos = body_start;

        match self.format_variant_list(enum_def, body_start, span.hi()) {
            Some(ref s) if enum_def.variants.is_empty() => self.push_str(s),
            rw => {
                self.push_rewrite(mk_sp(body_start, span.hi()), rw);
                self.block_indent = self.block_indent.block_unindent(self.config);
            }
        }
    }

    // Format the body of an enum definition
    fn format_variant_list(
        &mut self,
        enum_def: &ast::EnumDef,
        body_lo: BytePos,
        body_hi: BytePos,
    ) -> Option<String> {
        if enum_def.variants.is_empty() {
            let mut buffer = String::with_capacity(128);
            // 1 = "}"
            let span = mk_sp(body_lo, body_hi - BytePos(1));
            format_empty_struct_or_tuple(
                &self.get_context(),
                span,
                self.block_indent,
                &mut buffer,
                "",
                "}",
            );
            return Some(buffer);
        }
        let mut result = String::with_capacity(1024);
        let original_offset = self.block_indent;
        self.block_indent = self.block_indent.block_indent(self.config);

        // If enum variants have discriminants, try to vertically align those,
        // provided the discrims are not shifted too much  to the right
        let align_threshold: usize = self.config.enum_discrim_align_threshold();
        let discr_ident_lens: Vec<usize> = enum_def
            .variants
            .iter()
            .filter(|var| var.disr_expr.is_some())
            .map(|var| rewrite_ident(&self.get_context(), var.ident).len())
            .collect();
        // cut the list at the point of longest discrim shorter than the threshold
        // All of the discrims under the threshold will get padded, and all above - left as is.
        let pad_discrim_ident_to = *discr_ident_lens
            .iter()
            .filter(|&l| *l <= align_threshold)
            .max()
            .unwrap_or(&0);

        let itemize_list_with = |one_line_width: usize| {
            itemize_list(
                self.snippet_provider,
                enum_def.variants.iter(),
                "}",
                ",",
                |f| {
                    if !f.attrs.is_empty() {
                        f.attrs[0].span.lo()
                    } else {
                        f.span.lo()
                    }
                },
                |f| f.span.hi(),
                |f| {
                    self.format_variant(f, one_line_width, pad_discrim_ident_to)
                        .unknown_error()
                },
                body_lo,
                body_hi,
                false,
            )
            .collect()
        };
        let mut items: Vec<_> = itemize_list_with(self.config.struct_variant_width());

        // If one of the variants use multiple lines, use multi-lined formatting for all variants.
        let has_multiline_variant = items.iter().any(|item| item.inner_as_ref().contains('\n'));
        let has_single_line_variant = items.iter().any(|item| !item.inner_as_ref().contains('\n'));
        if has_multiline_variant && has_single_line_variant {
            items = itemize_list_with(0);
        }

        let shape = self.shape().sub_width(2)?;
        let fmt = ListFormatting::new(shape, self.config)
            .trailing_separator(self.config.trailing_comma())
            .preserve_newline(true);

        let list = write_list(&items, &fmt).ok()?;
        result.push_str(&list);
        result.push_str(&original_offset.to_string_with_newline(self.config));
        result.push('}');
        Some(result)
    }

    // Variant of an enum.
    fn format_variant(
        &self,
        field: &ast::Variant,
        one_line_width: usize,
        pad_discrim_ident_to: usize,
    ) -> Option<String> {
        if contains_skip(&field.attrs) {
            let lo = field.attrs[0].span.lo();
            let span = mk_sp(lo, field.span.hi());
            return Some(self.snippet(span).to_owned());
        }

        let context = self.get_context();
        let shape = self.shape();
        let attrs_str = if context.config.style_edition() >= StyleEdition::Edition2024 {
            field.attrs.rewrite(&context, shape)?
        } else {
            // StyleEdition::Edition20{15|18|21} formatting that was off by 1. See issue #5801
            field.attrs.rewrite(&context, shape.sub_width(1)?)?
        };
        // sub_width(1) to take the trailing comma into account
        let shape = shape.sub_width(1)?;

        let lo = field
            .attrs
            .last()
            .map_or(field.span.lo(), |attr| attr.span.hi());
        let span = mk_sp(lo, field.span.lo());

        let variant_body = match field.data {
            ast::VariantData::Tuple(..) | ast::VariantData::Struct { .. } => format_struct(
                &context,
                &StructParts::from_variant(field, &context),
                self.block_indent,
                Some(one_line_width),
            )?,
            ast::VariantData::Unit(..) => rewrite_ident(&context, field.ident).to_owned(),
        };

        let variant_body = if let Some(ref expr) = field.disr_expr {
            let lhs = format!("{variant_body:pad_discrim_ident_to$} =");
            let ex = &*expr.value;
            rewrite_assign_rhs_with(
                &context,
                lhs,
                ex,
                shape,
                &RhsAssignKind::Expr(&ex.kind, ex.span),
                RhsTactics::AllowOverflow,
            )
            .ok()?
        } else {
            variant_body
        };

        combine_strs_with_missing_comments(&context, &attrs_str, &variant_body, span, shape, false)
            .ok()
    }

    fn visit_impl_items(&mut self, items: &[Box<ast::AssocItem>]) {
        if self.get_context().config.reorder_impl_items() {
            type TyOpt = Option<Box<ast::Ty>>;
            use crate::ast::AssocItemKind::*;
            let is_type = |ty: &TyOpt| opaque_ty(ty).is_none();
            let is_opaque = |ty: &TyOpt| opaque_ty(ty).is_some();
            let both_type = |l: &TyOpt, r: &TyOpt| is_type(l) && is_type(r);
            let both_opaque = |l: &TyOpt, r: &TyOpt| is_opaque(l) && is_opaque(r);
            let need_empty_line = |a: &ast::AssocItemKind, b: &ast::AssocItemKind| match (a, b) {
                (Type(lty), Type(rty))
                    if both_type(&lty.ty, &rty.ty) || both_opaque(&lty.ty, &rty.ty) =>
                {
                    false
                }
                (Const(..), Const(..)) => false,
                _ => true,
            };

            // Create visitor for each items, then reorder them.
            let mut buffer = vec![];
            for item in items {
                self.visit_impl_item(item);
                buffer.push((self.buffer.clone(), item.clone()));
                self.buffer.clear();
            }

            buffer.sort_by(|(_, a), (_, b)| match (&a.kind, &b.kind) {
                (Type(lty), Type(rty))
                    if both_type(&lty.ty, &rty.ty) || both_opaque(&lty.ty, &rty.ty) =>
                {
                    lty.ident.as_str().cmp(rty.ident.as_str())
                }
                (Const(ca), Const(cb)) => ca.ident.as_str().cmp(cb.ident.as_str()),
                (MacCall(..), MacCall(..)) => Ordering::Equal,
                (Fn(..), Fn(..)) | (Delegation(..), Delegation(..)) => {
                    a.span.lo().cmp(&b.span.lo())
                }
                (Type(ty), _) if is_type(&ty.ty) => Ordering::Less,
                (_, Type(ty)) if is_type(&ty.ty) => Ordering::Greater,
                (Type(..), _) => Ordering::Less,
                (_, Type(..)) => Ordering::Greater,
                (Const(..), _) => Ordering::Less,
                (_, Const(..)) => Ordering::Greater,
                (MacCall(..), _) => Ordering::Less,
                (_, MacCall(..)) => Ordering::Greater,
                (Delegation(..), _) | (DelegationMac(..), _) => Ordering::Less,
                (_, Delegation(..)) | (_, DelegationMac(..)) => Ordering::Greater,
            });
            let mut prev_kind = None;
            for (buf, item) in buffer {
                // Make sure that there are at least a single empty line between
                // different impl items.
                if prev_kind
                    .as_ref()
                    .map_or(false, |prev_kind| need_empty_line(prev_kind, &item.kind))
                {
                    self.push_str("\n");
                }
                let indent_str = self.block_indent.to_string_with_newline(self.config);
                self.push_str(&indent_str);
                self.push_str(buf.trim());
                prev_kind = Some(item.kind.clone());
            }
        } else {
            for item in items {
                self.visit_impl_item(item);
            }
        }
    }
}

pub(crate) fn format_impl(
    context: &RewriteContext<'_>,
    item: &ast::Item,
    iimpl: &ast::Impl,
    offset: Indent,
) -> Option<String> {
    let ast::Impl {
        generics,
        self_ty,
        items,
        ..
    } = iimpl;
    let mut result = String::with_capacity(128);
    let ref_and_type = format_impl_ref_and_type(context, item, iimpl, offset)?;
    let sep = offset.to_string_with_newline(context.config);
    result.push_str(&ref_and_type);

    let where_budget = if result.contains('\n') {
        context.config.max_width()
    } else {
        context.budget(last_line_width(&result))
    };

    let mut option = WhereClauseOption::snuggled(&ref_and_type);
    let snippet = context.snippet(item.span);
    let open_pos = snippet.find_uncommented("{")? + 1;
    if !contains_comment(&snippet[open_pos..])
        && items.is_empty()
        && generics.where_clause.predicates.len() == 1
        && !result.contains('\n')
    {
        option.suppress_comma();
        option.snuggle();
        option.allow_single_line();
    }

    let missing_span = mk_sp(self_ty.span.hi(), item.span.hi());
    let where_span_end = context.snippet_provider.opt_span_before(missing_span, "{");
    let where_clause_str = rewrite_where_clause(
        context,
        &generics.where_clause.predicates,
        generics.where_clause.span,
        context.config.brace_style(),
        Shape::legacy(where_budget, offset.block_only()),
        false,
        "{",
        where_span_end,
        self_ty.span.hi(),
        option,
    )
    .ok()?;

    // If there is no where-clause, we may have missing comments between the trait name and
    // the opening brace.
    if generics.where_clause.predicates.is_empty() {
        if let Some(hi) = where_span_end {
            match recover_missing_comment_in_span(
                mk_sp(self_ty.span.hi(), hi),
                Shape::indented(offset, context.config),
                context,
                last_line_width(&result),
            ) {
                Ok(ref missing_comment) if !missing_comment.is_empty() => {
                    result.push_str(missing_comment);
                }
                _ => (),
            }
        }
    }

    if is_impl_single_line(context, items.as_slice(), &result, &where_clause_str, item)? {
        result.push_str(&where_clause_str);
        if where_clause_str.contains('\n') {
            // If there is only one where-clause predicate
            // and the where-clause spans multiple lines,
            // then recover the suppressed comma in single line where-clause formatting
            if generics.where_clause.predicates.len() == 1 {
                result.push(',');
            }
        }
        if where_clause_str.contains('\n') || last_line_contains_single_line_comment(&result) {
            result.push_str(&format!("{sep}{{{sep}}}"));
        } else {
            result.push_str(" {}");
        }
        return Some(result);
    }

    result.push_str(&where_clause_str);

    let need_newline = last_line_contains_single_line_comment(&result) || result.contains('\n');
    match context.config.brace_style() {
        _ if need_newline => result.push_str(&sep),
        BraceStyle::AlwaysNextLine => result.push_str(&sep),
        BraceStyle::PreferSameLine => result.push(' '),
        BraceStyle::SameLineWhere => {
            if !where_clause_str.is_empty() {
                result.push_str(&sep);
            } else {
                result.push(' ');
            }
        }
    }

    result.push('{');
    // this is an impl body snippet(impl SampleImpl { /* here */ })
    let lo = max(self_ty.span.hi(), generics.where_clause.span.hi());
    let snippet = context.snippet(mk_sp(lo, item.span.hi()));
    let open_pos = snippet.find_uncommented("{")? + 1;

    if !items.is_empty() || contains_comment(&snippet[open_pos..]) {
        let mut visitor = FmtVisitor::from_context(context);
        let item_indent = offset.block_only().block_indent(context.config);
        visitor.block_indent = item_indent;
        visitor.last_pos = lo + BytePos(open_pos as u32);

        visitor.visit_attrs(&item.attrs, ast::AttrStyle::Inner);
        visitor.visit_impl_items(items);

        visitor.format_missing(item.span.hi() - BytePos(1));

        let inner_indent_str = visitor.block_indent.to_string_with_newline(context.config);
        let outer_indent_str = offset.block_only().to_string_with_newline(context.config);

        result.push_str(&inner_indent_str);
        result.push_str(visitor.buffer.trim());
        result.push_str(&outer_indent_str);
    } else if need_newline || !context.config.empty_item_single_line() {
        result.push_str(&sep);
    }

    result.push('}');

    Some(result)
}

fn is_impl_single_line(
    context: &RewriteContext<'_>,
    items: &[Box<ast::AssocItem>],
    result: &str,
    where_clause_str: &str,
    item: &ast::Item,
) -> Option<bool> {
    let snippet = context.snippet(item.span);
    let open_pos = snippet.find_uncommented("{")? + 1;

    Some(
        context.config.empty_item_single_line()
            && items.is_empty()
            && !result.contains('\n')
            && result.len() + where_clause_str.len() <= context.config.max_width()
            && !contains_comment(&snippet[open_pos..]),
    )
}

fn format_impl_ref_and_type(
    context: &RewriteContext<'_>,
    item: &ast::Item,
    iimpl: &ast::Impl,
    offset: Indent,
) -> Option<String> {
    let ast::Impl {
        generics,
        of_trait,
        self_ty,
        items: _,
    } = iimpl;
    let mut result = String::with_capacity(128);

    result.push_str(&format_visibility(context, &item.vis));

    if let Some(of_trait) = of_trait.as_deref() {
        result.push_str(format_defaultness(of_trait.defaultness));
        result.push_str(format_safety(of_trait.safety));
    }

    let shape = if context.config.style_edition() >= StyleEdition::Edition2024 {
        Shape::indented(offset + last_line_width(&result), context.config)
    } else {
        generics_shape_from_config(
            context.config,
            Shape::indented(offset + last_line_width(&result), context.config),
            0,
        )?
    };
    let generics_str = rewrite_generics(context, "impl", generics, shape).ok()?;
    result.push_str(&generics_str);

    let trait_ref_overhead;
    if let Some(of_trait) = of_trait.as_deref() {
        result.push_str(format_constness_right(of_trait.constness));
        let polarity_str = match of_trait.polarity {
            ast::ImplPolarity::Negative(_) => "!",
            ast::ImplPolarity::Positive => "",
        };
        let result_len = last_line_width(&result);
        result.push_str(&rewrite_trait_ref(
            context,
            &of_trait.trait_ref,
            offset,
            polarity_str,
            result_len,
        )?);
        trait_ref_overhead = " for".len();
    } else {
        trait_ref_overhead = 0;
    }

    // Try to put the self type in a single line.
    let curly_brace_overhead = if generics.where_clause.predicates.is_empty() {
        // If there is no where-clause adapt budget for type formatting to take space and curly
        // brace into account.
        match context.config.brace_style() {
            BraceStyle::AlwaysNextLine => 0,
            _ => 2,
        }
    } else {
        0
    };
    let used_space = last_line_width(&result) + trait_ref_overhead + curly_brace_overhead;
    // 1 = space before the type.
    let budget = context.budget(used_space + 1);
    if let Some(self_ty_str) = self_ty.rewrite(context, Shape::legacy(budget, offset)) {
        if !self_ty_str.contains('\n') {
            if of_trait.is_some() {
                result.push_str(" for ");
            } else {
                result.push(' ');
            }
            result.push_str(&self_ty_str);
            return Some(result);
        }
    }

    // Couldn't fit the self type on a single line, put it on a new line.
    result.push('\n');
    // Add indentation of one additional tab.
    let new_line_offset = offset.block_indent(context.config);
    result.push_str(&new_line_offset.to_string(context.config));
    if of_trait.is_some() {
        result.push_str("for ");
    }
    let budget = context.budget(last_line_width(&result));
    let type_offset = match context.config.indent_style() {
        IndentStyle::Visual => new_line_offset + trait_ref_overhead,
        IndentStyle::Block => new_line_offset,
    };
    result.push_str(&*self_ty.rewrite(context, Shape::legacy(budget, type_offset))?);
    Some(result)
}

fn rewrite_trait_ref(
    context: &RewriteContext<'_>,
    trait_ref: &ast::TraitRef,
    offset: Indent,
    polarity_str: &str,
    result_len: usize,
) -> Option<String> {
    // 1 = space between generics and trait_ref
    let used_space = 1 + polarity_str.len() + result_len;
    let shape = Shape::indented(offset + used_space, context.config);
    if let Some(trait_ref_str) = trait_ref.rewrite(context, shape) {
        if !trait_ref_str.contains('\n') {
            return Some(format!(" {polarity_str}{trait_ref_str}"));
        }
    }
    // We could not make enough space for trait_ref, so put it on new line.
    let offset = offset.block_indent(context.config);
    let shape = Shape::indented(offset, context.config);
    let trait_ref_str = trait_ref.rewrite(context, shape)?;
    Some(format!(
        "{}{}{}",
        offset.to_string_with_newline(context.config),
        polarity_str,
        trait_ref_str
    ))
}

pub(crate) struct StructParts<'a> {
    prefix: &'a str,
    ident: symbol::Ident,
    vis: &'a ast::Visibility,
    def: &'a ast::VariantData,
    generics: Option<&'a ast::Generics>,
    span: Span,
}

impl<'a> StructParts<'a> {
    fn format_header(&self, context: &RewriteContext<'_>, offset: Indent) -> String {
        format_header(context, self.prefix, self.ident, self.vis, offset)
    }

    fn from_variant(variant: &'a ast::Variant, context: &RewriteContext<'_>) -> Self {
        StructParts {
            prefix: "",
            ident: variant.ident,
            vis: &DEFAULT_VISIBILITY,
            def: &variant.data,
            generics: None,
            span: enum_variant_span(variant, context),
        }
    }

    pub(crate) fn from_item(item: &'a ast::Item) -> Self {
        let (prefix, def, ident, generics) = match item.kind {
            ast::ItemKind::Struct(ident, ref generics, ref def) => {
                ("struct ", def, ident, generics)
            }
            ast::ItemKind::Union(ident, ref generics, ref def) => ("union ", def, ident, generics),
            _ => unreachable!(),
        };
        StructParts {
            prefix,
            ident,
            vis: &item.vis,
            def,
            generics: Some(generics),
            span: item.span,
        }
    }
}

fn enum_variant_span(variant: &ast::Variant, context: &RewriteContext<'_>) -> Span {
    use ast::VariantData::*;
    if let Some(ref anon_const) = variant.disr_expr {
        let span_before_consts = variant.span.until(anon_const.value.span);
        let hi = match &variant.data {
            Struct { .. } => context
                .snippet_provider
                .span_after_last(span_before_consts, "}"),
            Tuple(..) => context
                .snippet_provider
                .span_after_last(span_before_consts, ")"),
            Unit(..) => variant.ident.span.hi(),
        };
        mk_sp(span_before_consts.lo(), hi)
    } else {
        variant.span
    }
}

fn format_struct(
    context: &RewriteContext<'_>,
    struct_parts: &StructParts<'_>,
    offset: Indent,
    one_line_width: Option<usize>,
) -> Option<String> {
    match struct_parts.def {
        ast::VariantData::Unit(..) => format_unit_struct(context, struct_parts, offset),
        ast::VariantData::Tuple(fields, _) => {
            format_tuple_struct(context, struct_parts, fields, offset)
        }
        ast::VariantData::Struct { fields, .. } => {
            format_struct_struct(context, struct_parts, fields, offset, one_line_width)
        }
    }
}

pub(crate) fn format_trait(
    context: &RewriteContext<'_>,
    item: &ast::Item,
    offset: Indent,
) -> Option<String> {
    let ast::ItemKind::Trait(trait_kind) = &item.kind else {
        unreachable!();
    };
    let ast::Trait {
        constness,
        is_auto,
        safety,
        ident,
        ref generics,
        ref bounds,
        ref items,
    } = **trait_kind;

    let mut result = String::with_capacity(128);
    let header = format!(
        "{}{}{}{}trait ",
        format_visibility(context, &item.vis),
        format_constness(constness),
        format_safety(safety),
        format_auto(is_auto),
    );
    result.push_str(&header);

    let body_lo = context.snippet_provider.span_after(item.span, "{");

    let shape = Shape::indented(offset, context.config).offset_left(result.len())?;
    let generics_str =
        rewrite_generics(context, rewrite_ident(context, ident), generics, shape).ok()?;
    result.push_str(&generics_str);

    // FIXME(#2055): rustfmt fails to format when there are comments between trait bounds.
    if !bounds.is_empty() {
        // Retrieve *unnormalized* ident (See #6069)
        let source_ident = context.snippet(ident.span);
        let ident_hi = context.snippet_provider.span_after(item.span, source_ident);
        let bound_hi = bounds.last().unwrap().span().hi();
        let snippet = context.snippet(mk_sp(ident_hi, bound_hi));
        if contains_comment(snippet) {
            return None;
        }

        result = rewrite_assign_rhs_with(
            context,
            result + ":",
            bounds,
            shape,
            &RhsAssignKind::Bounds,
            RhsTactics::ForceNextLineWithoutIndent,
        )
        .ok()?;
    }

    // Rewrite where-clause.
    if !generics.where_clause.predicates.is_empty() {
        let where_on_new_line = context.config.indent_style() != IndentStyle::Block;

        let where_budget = context.budget(last_line_width(&result));
        let pos_before_where = if bounds.is_empty() {
            generics.where_clause.span.lo()
        } else {
            bounds[bounds.len() - 1].span().hi()
        };
        let option = WhereClauseOption::snuggled(&generics_str);
        let where_clause_str = rewrite_where_clause(
            context,
            &generics.where_clause.predicates,
            generics.where_clause.span,
            context.config.brace_style(),
            Shape::legacy(where_budget, offset.block_only()),
            where_on_new_line,
            "{",
            None,
            pos_before_where,
            option,
        )
        .ok()?;
        // If the where-clause cannot fit on the same line,
        // put the where-clause on a new line
        if !where_clause_str.contains('\n')
            && last_line_width(&result) + where_clause_str.len() + offset.width()
                > context.config.comment_width()
        {
            let width = offset.block_indent + context.config.tab_spaces() - 1;
            let where_indent = Indent::new(0, width);
            result.push_str(&where_indent.to_string_with_newline(context.config));
        }
        result.push_str(&where_clause_str);
    } else {
        let item_snippet = context.snippet(item.span);
        if let Some(lo) = item_snippet.find('/') {
            // 1 = `{`
            let comment_hi = if generics.params.len() > 0 {
                generics.span.lo() - BytePos(1)
            } else {
                body_lo - BytePos(1)
            };
            let comment_lo = item.span.lo() + BytePos(lo as u32);
            if comment_lo < comment_hi {
                match recover_missing_comment_in_span(
                    mk_sp(comment_lo, comment_hi),
                    Shape::indented(offset, context.config),
                    context,
                    last_line_width(&result),
                ) {
                    Ok(ref missing_comment) if !missing_comment.is_empty() => {
                        result.push_str(missing_comment);
                    }
                    _ => (),
                }
            }
        }
    }

    let block_span = mk_sp(generics.where_clause.span.hi(), item.span.hi());
    let snippet = context.snippet(block_span);
    let open_pos = snippet.find_uncommented("{")? + 1;

    match context.config.brace_style() {
        _ if last_line_contains_single_line_comment(&result)
            || last_line_width(&result) + 2 > context.budget(offset.width()) =>
        {
            result.push_str(&offset.to_string_with_newline(context.config));
        }
        _ if context.config.empty_item_single_line()
            && items.is_empty()
            && !result.contains('\n')
            && !contains_comment(&snippet[open_pos..]) =>
        {
            result.push_str(" {}");
            return Some(result);
        }
        BraceStyle::AlwaysNextLine => {
            result.push_str(&offset.to_string_with_newline(context.config));
        }
        BraceStyle::PreferSameLine => result.push(' '),
        BraceStyle::SameLineWhere => {
            if result.contains('\n')
                || (!generics.where_clause.predicates.is_empty() && !items.is_empty())
            {
                result.push_str(&offset.to_string_with_newline(context.config));
            } else {
                result.push(' ');
            }
        }
    }
    result.push('{');

    let outer_indent_str = offset.block_only().to_string_with_newline(context.config);

    if !items.is_empty() || contains_comment(&snippet[open_pos..]) {
        let mut visitor = FmtVisitor::from_context(context);
        visitor.block_indent = offset.block_only().block_indent(context.config);
        visitor.last_pos = block_span.lo() + BytePos(open_pos as u32);

        for item in items {
            visitor.visit_trait_item(item);
        }

        visitor.format_missing(item.span.hi() - BytePos(1));

        let inner_indent_str = visitor.block_indent.to_string_with_newline(context.config);

        result.push_str(&inner_indent_str);
        result.push_str(visitor.buffer.trim());
        result.push_str(&outer_indent_str);
    } else if result.contains('\n') {
        result.push_str(&outer_indent_str);
    }

    result.push('}');
    Some(result)
}

pub(crate) struct TraitAliasBounds<'a> {
    generic_bounds: &'a ast::GenericBounds,
    generics: &'a ast::Generics,
}

impl<'a> Rewrite for TraitAliasBounds<'a> {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        self.rewrite_result(context, shape).ok()
    }

    fn rewrite_result(&self, context: &RewriteContext<'_>, shape: Shape) -> RewriteResult {
        let generic_bounds_str = self.generic_bounds.rewrite_result(context, shape)?;

        let mut option = WhereClauseOption::new(true, WhereClauseSpace::None);
        option.allow_single_line();

        let where_str = rewrite_where_clause(
            context,
            &self.generics.where_clause.predicates,
            self.generics.where_clause.span,
            context.config.brace_style(),
            shape,
            false,
            ";",
            None,
            self.generics.where_clause.span.lo(),
            option,
        )?;

        let fits_single_line = !generic_bounds_str.contains('\n')
            && !where_str.contains('\n')
            && generic_bounds_str.len() + where_str.len() < shape.width;
        let space = if generic_bounds_str.is_empty() || where_str.is_empty() {
            Cow::from("")
        } else if fits_single_line {
            Cow::from(" ")
        } else {
            shape.indent.to_string_with_newline(context.config)
        };

        Ok(format!("{generic_bounds_str}{space}{where_str}"))
    }
}

pub(crate) fn format_trait_alias(
    context: &RewriteContext<'_>,
    ident: symbol::Ident,
    vis: &ast::Visibility,
    generics: &ast::Generics,
    generic_bounds: &ast::GenericBounds,
    shape: Shape,
) -> Option<String> {
    let alias = rewrite_ident(context, ident);
    // 6 = "trait ", 2 = " ="
    let g_shape = shape.offset_left(6)?.sub_width(2)?;
    let generics_str = rewrite_generics(context, alias, generics, g_shape).ok()?;
    let vis_str = format_visibility(context, vis);
    let lhs = format!("{vis_str}trait {generics_str} =");
    // 1 = ";"
    let trait_alias_bounds = TraitAliasBounds {
        generic_bounds,
        generics,
    };
    rewrite_assign_rhs(
        context,
        lhs,
        &trait_alias_bounds,
        &RhsAssignKind::Bounds,
        shape.sub_width(1)?,
    )
    .map(|s| s + ";")
    .ok()
}

fn format_unit_struct(
    context: &RewriteContext<'_>,
    p: &StructParts<'_>,
    offset: Indent,
) -> Option<String> {
    let header_str = format_header(context, p.prefix, p.ident, p.vis, offset);
    let generics_str = if let Some(generics) = p.generics {
        let hi = context.snippet_provider.span_before_last(p.span, ";");
        format_generics(
            context,
            generics,
            context.config.brace_style(),
            BracePos::None,
            offset,
            // make a span that starts right after `struct Foo`
            mk_sp(p.ident.span.hi(), hi),
            last_line_width(&header_str),
        )?
    } else {
        String::new()
    };
    Some(format!("{header_str}{generics_str};"))
}

pub(crate) fn format_struct_struct(
    context: &RewriteContext<'_>,
    struct_parts: &StructParts<'_>,
    fields: &[ast::FieldDef],
    offset: Indent,
    one_line_width: Option<usize>,
) -> Option<String> {
    let mut result = String::with_capacity(1024);
    let span = struct_parts.span;

    let header_str = struct_parts.format_header(context, offset);
    result.push_str(&header_str);

    let header_hi = struct_parts.ident.span.hi();
    let body_lo = if let Some(generics) = struct_parts.generics {
        // Adjust the span to start at the end of the generic arguments before searching for the '{'
        let span = span.with_lo(generics.where_clause.span.hi());
        context.snippet_provider.span_after(span, "{")
    } else {
        context.snippet_provider.span_after(span, "{")
    };

    let generics_str = match struct_parts.generics {
        Some(g) => format_generics(
            context,
            g,
            context.config.brace_style(),
            if fields.is_empty() {
                BracePos::ForceSameLine
            } else {
                BracePos::Auto
            },
            offset,
            // make a span that starts right after `struct Foo`
            mk_sp(header_hi, body_lo),
            last_line_width(&result),
        )?,
        None => {
            // 3 = ` {}`, 2 = ` {`.
            let overhead = if fields.is_empty() { 3 } else { 2 };
            if (context.config.brace_style() == BraceStyle::AlwaysNextLine && !fields.is_empty())
                || context.config.max_width() < overhead + result.len()
            {
                format!("\n{}{{", offset.block_only().to_string(context.config))
            } else {
                " {".to_owned()
            }
        }
    };
    // 1 = `}`
    let overhead = if fields.is_empty() { 1 } else { 0 };
    let total_width = result.len() + generics_str.len() + overhead;
    if !generics_str.is_empty()
        && !generics_str.contains('\n')
        && total_width > context.config.max_width()
    {
        result.push('\n');
        result.push_str(&offset.to_string(context.config));
        result.push_str(generics_str.trim_start());
    } else {
        result.push_str(&generics_str);
    }

    if fields.is_empty() {
        let inner_span = mk_sp(body_lo, span.hi() - BytePos(1));
        format_empty_struct_or_tuple(context, inner_span, offset, &mut result, "", "}");
        return Some(result);
    }

    // 3 = ` ` and ` }`
    let one_line_budget = context.budget(result.len() + 3 + offset.width());
    let one_line_budget =
        one_line_width.map_or(0, |one_line_width| min(one_line_width, one_line_budget));

    let items_str = rewrite_with_alignment(
        fields,
        context,
        Shape::indented(offset.block_indent(context.config), context.config).sub_width(1)?,
        mk_sp(body_lo, span.hi()),
        one_line_budget,
    )?;

    if !items_str.contains('\n')
        && !result.contains('\n')
        && items_str.len() <= one_line_budget
        && !last_line_contains_single_line_comment(&items_str)
    {
        Some(format!("{result} {items_str} }}"))
    } else {
        Some(format!(
            "{}\n{}{}\n{}}}",
            result,
            offset
                .block_indent(context.config)
                .to_string(context.config),
            items_str,
            offset.to_string(context.config)
        ))
    }
}

fn get_bytepos_after_visibility(vis: &ast::Visibility, default_span: Span) -> BytePos {
    match vis.kind {
        ast::VisibilityKind::Restricted { .. } => vis.span.hi(),
        _ => default_span.lo(),
    }
}

// Format tuple or struct without any fields. We need to make sure that the comments
// inside the delimiters are preserved.
fn format_empty_struct_or_tuple(
    context: &RewriteContext<'_>,
    span: Span,
    offset: Indent,
    result: &mut String,
    opener: &str,
    closer: &str,
) {
    // 3 = " {}" or "();"
    let used_width = last_line_used_width(result, offset.width()) + 3;
    if used_width > context.config.max_width() {
        result.push_str(&offset.to_string_with_newline(context.config))
    }
    result.push_str(opener);

    // indented shape for proper indenting of multi-line comments
    let shape = Shape::indented(offset.block_indent(context.config), context.config);
    match rewrite_missing_comment(span, shape, context) {
        Ok(ref s) if s.is_empty() => (),
        Ok(ref s) => {
            let is_multi_line = !is_single_line(s);
            if is_multi_line || first_line_contains_single_line_comment(s) {
                let nested_indent_str = offset
                    .block_indent(context.config)
                    .to_string_with_newline(context.config);
                result.push_str(&nested_indent_str);
            }
            result.push_str(s);
            if is_multi_line || last_line_contains_single_line_comment(s) {
                result.push_str(&offset.to_string_with_newline(context.config));
            }
        }
        Err(_) => result.push_str(context.snippet(span)),
    }
    result.push_str(closer);
}

fn format_tuple_struct(
    context: &RewriteContext<'_>,
    struct_parts: &StructParts<'_>,
    fields: &[ast::FieldDef],
    offset: Indent,
) -> Option<String> {
    let mut result = String::with_capacity(1024);
    let span = struct_parts.span;

    let header_str = struct_parts.format_header(context, offset);
    result.push_str(&header_str);

    let body_lo = if fields.is_empty() {
        let lo = get_bytepos_after_visibility(struct_parts.vis, span);
        context
            .snippet_provider
            .span_after(mk_sp(lo, span.hi()), "(")
    } else {
        fields[0].span.lo()
    };
    let body_hi = if fields.is_empty() {
        context
            .snippet_provider
            .span_after(mk_sp(body_lo, span.hi()), ")")
    } else {
        // This is a dirty hack to work around a missing `)` from the span of the last field.
        let last_arg_span = fields[fields.len() - 1].span;
        context
            .snippet_provider
            .opt_span_after(mk_sp(last_arg_span.hi(), span.hi()), ")")
            .unwrap_or_else(|| last_arg_span.hi())
    };

    let where_clause_str = match struct_parts.generics {
        Some(generics) => {
            let budget = context.budget(last_line_width(&header_str));
            let shape = Shape::legacy(budget, offset);
            let generics_str = rewrite_generics(context, "", generics, shape).ok()?;
            result.push_str(&generics_str);

            let where_budget = context.budget(last_line_width(&result));
            let option = WhereClauseOption::new(true, WhereClauseSpace::Newline);
            rewrite_where_clause(
                context,
                &generics.where_clause.predicates,
                generics.where_clause.span,
                context.config.brace_style(),
                Shape::legacy(where_budget, offset.block_only()),
                false,
                ";",
                None,
                body_hi,
                option,
            )
            .ok()?
        }
        None => "".to_owned(),
    };

    if fields.is_empty() {
        let body_hi = context
            .snippet_provider
            .span_before(mk_sp(body_lo, span.hi()), ")");
        let inner_span = mk_sp(body_lo, body_hi);
        format_empty_struct_or_tuple(context, inner_span, offset, &mut result, "(", ")");
    } else {
        let shape = Shape::indented(offset, context.config).sub_width(1)?;
        let lo = if let Some(generics) = struct_parts.generics {
            generics.span.hi()
        } else {
            struct_parts.ident.span.hi()
        };
        result = overflow::rewrite_with_parens(
            context,
            &result,
            fields.iter(),
            shape,
            mk_sp(lo, span.hi()),
            context.config.fn_call_width(),
            None,
        )
        .ok()?;
    }

    if !where_clause_str.is_empty()
        && !where_clause_str.contains('\n')
        && (result.contains('\n')
            || offset.block_indent + result.len() + where_clause_str.len() + 1
                > context.config.max_width())
    {
        // We need to put the where-clause on a new line, but we didn't
        // know that earlier, so the where-clause will not be indented properly.
        result.push('\n');
        result.push_str(
            &(offset.block_only() + (context.config.tab_spaces() - 1)).to_string(context.config),
        );
    }
    result.push_str(&where_clause_str);

    Some(result)
}

#[derive(Clone, Copy)]
pub(crate) enum ItemVisitorKind {
    Item,
    AssocTraitItem,
    AssocImplItem,
    ForeignItem,
}

struct TyAliasRewriteInfo<'c, 'g>(
    &'c RewriteContext<'c>,
    Indent,
    &'g ast::Generics,
    ast::TyAliasWhereClauses,
    symbol::Ident,
    Span,
);

pub(crate) fn rewrite_type_alias<'a>(
    ty_alias_kind: &ast::TyAlias,
    vis: &ast::Visibility,
    context: &RewriteContext<'a>,
    indent: Indent,
    visitor_kind: ItemVisitorKind,
    span: Span,
) -> RewriteResult {
    use ItemVisitorKind::*;

    let ast::TyAlias {
        defaultness,
        ident,
        ref generics,
        ref bounds,
        ref ty,
        where_clauses,
    } = *ty_alias_kind;
    let ty_opt = ty.as_ref();
    let rhs_hi = ty
        .as_ref()
        .map_or(where_clauses.before.span.hi(), |ty| ty.span.hi());
    let rw_info = &TyAliasRewriteInfo(context, indent, generics, where_clauses, ident, span);
    let op_ty = opaque_ty(ty);
    // Type Aliases are formatted slightly differently depending on the context
    // in which they appear, whether they are opaque, and whether they are associated.
    // https://rustc-dev-guide.rust-lang.org/opaque-types-type-alias-impl-trait.html
    // https://github.com/rust-dev-tools/fmt-rfcs/blob/master/guide/items.md#type-aliases
    match (visitor_kind, &op_ty) {
        (Item | AssocTraitItem | ForeignItem, Some(op_bounds)) => {
            let op = OpaqueType { bounds: op_bounds };
            rewrite_ty(rw_info, Some(bounds), Some(&op), rhs_hi, vis)
        }
        (Item | AssocTraitItem | ForeignItem, None) => {
            rewrite_ty(rw_info, Some(bounds), ty_opt, rhs_hi, vis)
        }
        (AssocImplItem, _) => {
            let result = if let Some(op_bounds) = op_ty {
                let op = OpaqueType { bounds: op_bounds };
                rewrite_ty(
                    rw_info,
                    Some(bounds),
                    Some(&op),
                    rhs_hi,
                    &DEFAULT_VISIBILITY,
                )
            } else {
                rewrite_ty(rw_info, Some(bounds), ty_opt, rhs_hi, vis)
            }?;
            match defaultness {
                ast::Defaultness::Default(..) => Ok(format!("default {result}")),
                _ => Ok(result),
            }
        }
    }
}

fn rewrite_ty<R: Rewrite>(
    rw_info: &TyAliasRewriteInfo<'_, '_>,
    generic_bounds_opt: Option<&ast::GenericBounds>,
    rhs: Option<&R>,
    // the span of the end of the RHS (or the end of the generics, if there is no RHS)
    rhs_hi: BytePos,
    vis: &ast::Visibility,
) -> RewriteResult {
    let mut result = String::with_capacity(128);
    let TyAliasRewriteInfo(context, indent, generics, where_clauses, ident, span) = *rw_info;
    let (before_where_predicates, after_where_predicates) = generics
        .where_clause
        .predicates
        .split_at(where_clauses.split);
    result.push_str(&format!("{}type ", format_visibility(context, vis)));
    let ident_str = rewrite_ident(context, ident);

    if generics.params.is_empty() {
        result.push_str(ident_str)
    } else {
        // 2 = `= `
        let g_shape = Shape::indented(indent, context.config);
        let g_shape = g_shape
            .offset_left(result.len())
            .and_then(|s| s.sub_width(2))
            .max_width_error(g_shape.width, span)?;
        let generics_str = rewrite_generics(context, ident_str, generics, g_shape)?;
        result.push_str(&generics_str);
    }

    if let Some(bounds) = generic_bounds_opt {
        if !bounds.is_empty() {
            // 2 = `: `
            let shape = Shape::indented(indent, context.config);
            let shape = shape
                .offset_left(result.len() + 2)
                .max_width_error(shape.width, span)?;
            let type_bounds = bounds
                .rewrite_result(context, shape)
                .map(|s| format!(": {}", s))?;
            result.push_str(&type_bounds);
        }
    }

    let where_budget = context.budget(last_line_width(&result));
    let mut option = WhereClauseOption::snuggled(&result);
    if rhs.is_none() {
        option.suppress_comma();
    }
    let before_where_clause_str = rewrite_where_clause(
        context,
        before_where_predicates,
        where_clauses.before.span,
        context.config.brace_style(),
        Shape::legacy(where_budget, indent),
        false,
        "=",
        None,
        generics.span.hi(),
        option,
    )?;
    result.push_str(&before_where_clause_str);

    let mut result = if let Some(ty) = rhs {
        // If there are any where clauses, add a newline before the assignment.
        // If there is a before where clause, do not indent, but if there is
        // only an after where clause, additionally indent the type.
        if !before_where_predicates.is_empty() {
            result.push_str(&indent.to_string_with_newline(context.config));
        } else if !after_where_predicates.is_empty() {
            result.push_str(
                &indent
                    .block_indent(context.config)
                    .to_string_with_newline(context.config),
            );
        } else {
            result.push(' ');
        }

        let comment_span = context
            .snippet_provider
            .opt_span_before(span, "=")
            .map(|op_lo| mk_sp(where_clauses.before.span.hi(), op_lo));

        let lhs = match comment_span {
            Some(comment_span)
                if contains_comment(
                    context
                        .snippet_provider
                        .span_to_snippet(comment_span)
                        .unknown_error()?,
                ) =>
            {
                let comment_shape = if !before_where_predicates.is_empty() {
                    Shape::indented(indent, context.config)
                } else {
                    let shape = Shape::indented(indent, context.config);
                    shape
                        .block_left(context.config.tab_spaces())
                        .max_width_error(shape.width, span)?
                };

                combine_strs_with_missing_comments(
                    context,
                    result.trim_end(),
                    "=",
                    comment_span,
                    comment_shape,
                    true,
                )?
            }
            _ => format!("{result}="),
        };

        // 1 = `;` unless there's a trailing where clause
        let shape = Shape::indented(indent, context.config);
        let shape = if after_where_predicates.is_empty() {
            Shape::indented(indent, context.config)
                .sub_width(1)
                .max_width_error(shape.width, span)?
        } else {
            shape
        };
        rewrite_assign_rhs(context, lhs, &*ty, &RhsAssignKind::Ty, shape)?
    } else {
        result
    };

    if !after_where_predicates.is_empty() {
        let option = WhereClauseOption::new(true, WhereClauseSpace::Newline);
        let after_where_clause_str = rewrite_where_clause(
            context,
            after_where_predicates,
            where_clauses.after.span,
            context.config.brace_style(),
            Shape::indented(indent, context.config),
            false,
            ";",
            None,
            rhs_hi,
            option,
        )?;
        result.push_str(&after_where_clause_str);
    }

    result += ";";
    Ok(result)
}

fn type_annotation_spacing(config: &Config) -> (&str, &str) {
    (
        if config.space_before_colon() { " " } else { "" },
        if config.space_after_colon() { " " } else { "" },
    )
}

pub(crate) fn rewrite_struct_field_prefix(
    context: &RewriteContext<'_>,
    field: &ast::FieldDef,
) -> RewriteResult {
    let vis = format_visibility(context, &field.vis);
    let safety = format_safety(field.safety);
    let type_annotation_spacing = type_annotation_spacing(context.config);
    Ok(match field.ident {
        Some(name) => format!(
            "{vis}{safety}{}{}:",
            rewrite_ident(context, name),
            type_annotation_spacing.0
        ),
        None => format!("{vis}{safety}"),
    })
}

impl Rewrite for ast::FieldDef {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        self.rewrite_result(context, shape).ok()
    }

    fn rewrite_result(&self, context: &RewriteContext<'_>, shape: Shape) -> RewriteResult {
        rewrite_struct_field(context, self, shape, 0)
    }
}

pub(crate) fn rewrite_struct_field(
    context: &RewriteContext<'_>,
    field: &ast::FieldDef,
    shape: Shape,
    lhs_max_width: usize,
) -> RewriteResult {
    // FIXME(default_field_values): Implement formatting.
    if field.default.is_some() {
        return Err(RewriteError::Unknown);
    }

    if contains_skip(&field.attrs) {
        return Ok(context.snippet(field.span()).to_owned());
    }

    let type_annotation_spacing = type_annotation_spacing(context.config);
    let prefix = rewrite_struct_field_prefix(context, field)?;

    let attrs_str = field.attrs.rewrite_result(context, shape)?;
    let attrs_extendable = field.ident.is_none() && is_attributes_extendable(&attrs_str);
    let missing_span = if field.attrs.is_empty() {
        mk_sp(field.span.lo(), field.span.lo())
    } else {
        mk_sp(field.attrs.last().unwrap().span.hi(), field.span.lo())
    };
    let mut spacing = String::from(if field.ident.is_some() {
        type_annotation_spacing.1
    } else {
        ""
    });
    // Try to put everything on a single line.
    let attr_prefix = combine_strs_with_missing_comments(
        context,
        &attrs_str,
        &prefix,
        missing_span,
        shape,
        attrs_extendable,
    )?;
    let overhead = trimmed_last_line_width(&attr_prefix);
    let lhs_offset = lhs_max_width.saturating_sub(overhead);
    for _ in 0..lhs_offset {
        spacing.push(' ');
    }
    // In this extreme case we will be missing a space between an attribute and a field.
    if prefix.is_empty() && !attrs_str.is_empty() && attrs_extendable && spacing.is_empty() {
        spacing.push(' ');
    }

    let orig_ty = shape
        .offset_left(overhead + spacing.len())
        .and_then(|ty_shape| field.ty.rewrite_result(context, ty_shape).ok());

    if let Some(ref ty) = orig_ty {
        if !ty.contains('\n') && !contains_comment(context.snippet(missing_span)) {
            return Ok(attr_prefix + &spacing + ty);
        }
    }

    let is_prefix_empty = prefix.is_empty();
    // We must use multiline. We are going to put attributes and a field on different lines.
    let field_str = rewrite_assign_rhs(context, prefix, &*field.ty, &RhsAssignKind::Ty, shape)?;
    // Remove a leading white-space from `rewrite_assign_rhs()` when rewriting a tuple struct.
    let field_str = if is_prefix_empty {
        field_str.trim_start()
    } else {
        &field_str
    };
    combine_strs_with_missing_comments(context, &attrs_str, field_str, missing_span, shape, false)
}

pub(crate) struct StaticParts<'a> {
    prefix: &'a str,
    safety: ast::Safety,
    vis: &'a ast::Visibility,
    ident: symbol::Ident,
    generics: Option<&'a ast::Generics>,
    ty: &'a ast::Ty,
    mutability: ast::Mutability,
    expr_opt: Option<&'a Box<ast::Expr>>,
    defaultness: Option<ast::Defaultness>,
    span: Span,
}

impl<'a> StaticParts<'a> {
    pub(crate) fn from_item(item: &'a ast::Item) -> Self {
        let (defaultness, prefix, safety, ident, ty, mutability, expr, generics) = match &item.kind
        {
            ast::ItemKind::Static(s) => (
                None,
                "static",
                s.safety,
                s.ident,
                &s.ty,
                s.mutability,
                &s.expr,
                None,
            ),
            ast::ItemKind::Const(c) => (
                Some(c.defaultness),
                "const",
                ast::Safety::Default,
                c.ident,
                &c.ty,
                ast::Mutability::Not,
                &c.expr,
                Some(&c.generics),
            ),
            _ => unreachable!(),
        };
        StaticParts {
            prefix,
            safety,
            vis: &item.vis,
            ident,
            generics,
            ty,
            mutability,
            expr_opt: expr.as_ref(),
            defaultness,
            span: item.span,
        }
    }

    pub(crate) fn from_trait_item(ti: &'a ast::AssocItem, ident: Ident) -> Self {
        let (defaultness, ty, expr_opt, generics) = match &ti.kind {
            ast::AssocItemKind::Const(c) => (c.defaultness, &c.ty, &c.expr, Some(&c.generics)),
            _ => unreachable!(),
        };
        StaticParts {
            prefix: "const",
            safety: ast::Safety::Default,
            vis: &ti.vis,
            ident,
            generics,
            ty,
            mutability: ast::Mutability::Not,
            expr_opt: expr_opt.as_ref(),
            defaultness: Some(defaultness),
            span: ti.span,
        }
    }

    pub(crate) fn from_impl_item(ii: &'a ast::AssocItem, ident: Ident) -> Self {
        let (defaultness, ty, expr, generics) = match &ii.kind {
            ast::AssocItemKind::Const(c) => (c.defaultness, &c.ty, &c.expr, Some(&c.generics)),
            _ => unreachable!(),
        };
        StaticParts {
            prefix: "const",
            safety: ast::Safety::Default,
            vis: &ii.vis,
            ident,
            generics,
            ty,
            mutability: ast::Mutability::Not,
            expr_opt: expr.as_ref(),
            defaultness: Some(defaultness),
            span: ii.span,
        }
    }
}

fn rewrite_static(
    context: &RewriteContext<'_>,
    static_parts: &StaticParts<'_>,
    offset: Indent,
) -> Option<String> {
    // For now, if this static (or const) has generics, then bail.
    if static_parts
        .generics
        .is_some_and(|g| !g.params.is_empty() || !g.where_clause.is_empty())
    {
        return None;
    }

    let colon = colon_spaces(context.config);
    let mut prefix = format!(
        "{}{}{}{} {}{}{}",
        format_visibility(context, static_parts.vis),
        static_parts.defaultness.map_or("", format_defaultness),
        format_safety(static_parts.safety),
        static_parts.prefix,
        format_mutability(static_parts.mutability),
        rewrite_ident(context, static_parts.ident),
        colon,
    );
    // 2 = " =".len()
    let ty_shape =
        Shape::indented(offset.block_only(), context.config).offset_left(prefix.len() + 2)?;
    let ty_str = match static_parts.ty.rewrite(context, ty_shape) {
        Some(ty_str) => ty_str,
        None => {
            if prefix.ends_with(' ') {
                prefix.pop();
            }
            let nested_indent = offset.block_indent(context.config);
            let nested_shape = Shape::indented(nested_indent, context.config);
            let ty_str = static_parts.ty.rewrite(context, nested_shape)?;
            format!(
                "{}{}",
                nested_indent.to_string_with_newline(context.config),
                ty_str
            )
        }
    };

    if let Some(expr) = static_parts.expr_opt {
        let comments_lo = context.snippet_provider.span_after(static_parts.span, "=");
        let expr_lo = expr.span.lo();
        let comments_span = mk_sp(comments_lo, expr_lo);

        let lhs = format!("{prefix}{ty_str} =");

        // 1 = ;
        let remaining_width = context.budget(offset.block_indent + 1);
        rewrite_assign_rhs_with_comments(
            context,
            &lhs,
            &**expr,
            Shape::legacy(remaining_width, offset.block_only()),
            &RhsAssignKind::Expr(&expr.kind, expr.span),
            RhsTactics::Default,
            comments_span,
            true,
        )
        .ok()
        .map(|res| recover_comment_removed(res, static_parts.span, context))
        .map(|s| if s.ends_with(';') { s } else { s + ";" })
    } else {
        Some(format!("{prefix}{ty_str};"))
    }
}

// FIXME(calebcartwright) - This is a hack around a bug in the handling of TyKind::ImplTrait.
// This should be removed once that bug is resolved, with the type alias formatting using the
// defined Ty for the RHS directly.
// https://github.com/rust-lang/rustfmt/issues/4373
// https://github.com/rust-lang/rustfmt/issues/5027
struct OpaqueType<'a> {
    bounds: &'a ast::GenericBounds,
}

impl<'a> Rewrite for OpaqueType<'a> {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        let shape = shape.offset_left(5)?; // `impl `
        self.bounds
            .rewrite(context, shape)
            .map(|s| format!("impl {}", s))
    }
}

impl Rewrite for ast::FnRetTy {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        self.rewrite_result(context, shape).ok()
    }

    fn rewrite_result(&self, context: &RewriteContext<'_>, shape: Shape) -> RewriteResult {
        match *self {
            ast::FnRetTy::Default(_) => Ok(String::new()),
            ast::FnRetTy::Ty(ref ty) => {
                let arrow_width = "-> ".len();
                if context.config.style_edition() <= StyleEdition::Edition2021
                    || context.config.indent_style() == IndentStyle::Visual
                {
                    let inner_width = shape
                        .width
                        .checked_sub(arrow_width)
                        .max_width_error(shape.width, self.span())?;
                    return ty
                        .rewrite_result(
                            context,
                            Shape::legacy(inner_width, shape.indent + arrow_width),
                        )
                        .map(|r| format!("-> {}", r));
                }

                let shape = shape
                    .offset_left(arrow_width)
                    .max_width_error(shape.width, self.span())?;

                ty.rewrite_result(context, shape)
                    .map(|s| format!("-> {}", s))
            }
        }
    }
}

fn is_empty_infer(ty: &ast::Ty, pat_span: Span) -> bool {
    match ty.kind {
        ast::TyKind::Infer => ty.span.hi() == pat_span.hi(),
        _ => false,
    }
}

/// Recover any missing comments between the param and the type.
///
/// # Returns
///
/// A 2-len tuple with the comment before the colon in first position, and the comment after the
/// colon in second position.
fn get_missing_param_comments(
    context: &RewriteContext<'_>,
    pat_span: Span,
    ty_span: Span,
    shape: Shape,
) -> (String, String) {
    let missing_comment_span = mk_sp(pat_span.hi(), ty_span.lo());

    let span_before_colon = {
        let missing_comment_span_hi = context
            .snippet_provider
            .span_before(missing_comment_span, ":");
        mk_sp(pat_span.hi(), missing_comment_span_hi)
    };
    let span_after_colon = {
        let missing_comment_span_lo = context
            .snippet_provider
            .span_after(missing_comment_span, ":");
        mk_sp(missing_comment_span_lo, ty_span.lo())
    };

    let comment_before_colon = rewrite_missing_comment(span_before_colon, shape, context)
        .ok()
        .filter(|comment| !comment.is_empty())
        .map_or(String::new(), |comment| format!(" {}", comment));
    let comment_after_colon = rewrite_missing_comment(span_after_colon, shape, context)
        .ok()
        .filter(|comment| !comment.is_empty())
        .map_or(String::new(), |comment| format!("{} ", comment));
    (comment_before_colon, comment_after_colon)
}

impl Rewrite for ast::Param {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        self.rewrite_result(context, shape).ok()
    }

    fn rewrite_result(&self, context: &RewriteContext<'_>, shape: Shape) -> RewriteResult {
        let param_attrs_result = self
            .attrs
            .rewrite_result(context, Shape::legacy(shape.width, shape.indent))?;
        // N.B. Doc comments aren't typically valid syntax, but could appear
        // in the presence of certain macros - https://github.com/rust-lang/rustfmt/issues/4936
        let (span, has_multiple_attr_lines, has_doc_comments) = if !self.attrs.is_empty() {
            let num_attrs = self.attrs.len();
            (
                mk_sp(self.attrs[num_attrs - 1].span.hi(), self.pat.span.lo()),
                param_attrs_result.contains('\n'),
                self.attrs.iter().any(|a| a.is_doc_comment()),
            )
        } else {
            (mk_sp(self.span.lo(), self.span.lo()), false, false)
        };

        if let Some(ref explicit_self) = self.to_self() {
            rewrite_explicit_self(
                context,
                explicit_self,
                &param_attrs_result,
                span,
                shape,
                has_multiple_attr_lines,
            )
        } else if is_named_param(self) {
            let param_name = &self
                .pat
                .rewrite_result(context, Shape::legacy(shape.width, shape.indent))?;
            let mut result = combine_strs_with_missing_comments(
                context,
                &param_attrs_result,
                param_name,
                span,
                shape,
                !has_multiple_attr_lines && !has_doc_comments,
            )?;

            if !is_empty_infer(&*self.ty, self.pat.span) {
                let (before_comment, after_comment) =
                    get_missing_param_comments(context, self.pat.span, self.ty.span, shape);
                result.push_str(&before_comment);
                result.push_str(colon_spaces(context.config));
                result.push_str(&after_comment);
                let overhead = last_line_width(&result);
                let max_width = shape
                    .width
                    .checked_sub(overhead)
                    .max_width_error(shape.width, self.span())?;
                if let Ok(ty_str) = self
                    .ty
                    .rewrite_result(context, Shape::legacy(max_width, shape.indent))
                {
                    result.push_str(&ty_str);
                } else {
                    let prev_str = if param_attrs_result.is_empty() {
                        param_attrs_result
                    } else {
                        param_attrs_result + &shape.to_string_with_newline(context.config)
                    };

                    result = combine_strs_with_missing_comments(
                        context,
                        &prev_str,
                        param_name,
                        span,
                        shape,
                        !has_multiple_attr_lines,
                    )?;
                    result.push_str(&before_comment);
                    result.push_str(colon_spaces(context.config));
                    result.push_str(&after_comment);
                    let overhead = last_line_width(&result);
                    let max_width = shape
                        .width
                        .checked_sub(overhead)
                        .max_width_error(shape.width, self.span())?;
                    let ty_str = self
                        .ty
                        .rewrite_result(context, Shape::legacy(max_width, shape.indent))?;
                    result.push_str(&ty_str);
                }
            }

            Ok(result)
        } else {
            self.ty.rewrite_result(context, shape)
        }
    }
}

fn rewrite_opt_lifetime(
    context: &RewriteContext<'_>,
    lifetime: Option<ast::Lifetime>,
) -> RewriteResult {
    let Some(l) = lifetime else {
        return Ok(String::new());
    };
    let mut result = l.rewrite_result(
        context,
        Shape::legacy(context.config.max_width(), Indent::empty()),
    )?;
    result.push(' ');
    Ok(result)
}

fn rewrite_explicit_self(
    context: &RewriteContext<'_>,
    explicit_self: &ast::ExplicitSelf,
    param_attrs: &str,
    span: Span,
    shape: Shape,
    has_multiple_attr_lines: bool,
) -> RewriteResult {
    let self_str = match explicit_self.node {
        ast::SelfKind::Region(lt, m) => {
            let mut_str = format_mutability(m);
            let lifetime_str = rewrite_opt_lifetime(context, lt)?;
            format!("&{lifetime_str}{mut_str}self")
        }
        ast::SelfKind::Pinned(lt, m) => {
            let mut_str = m.ptr_str();
            let lifetime_str = rewrite_opt_lifetime(context, lt)?;
            format!("&{lifetime_str}pin {mut_str} self")
        }
        ast::SelfKind::Explicit(ref ty, mutability) => {
            let type_str = ty.rewrite_result(
                context,
                Shape::legacy(context.config.max_width(), Indent::empty()),
            )?;
            format!("{}self: {}", format_mutability(mutability), type_str)
        }
        ast::SelfKind::Value(mutability) => format!("{}self", format_mutability(mutability)),
    };
    Ok(combine_strs_with_missing_comments(
        context,
        param_attrs,
        &self_str,
        span,
        shape,
        !has_multiple_attr_lines,
    )?)
}

pub(crate) fn span_lo_for_param(param: &ast::Param) -> BytePos {
    if param.attrs.is_empty() {
        if is_named_param(param) {
            param.pat.span.lo()
        } else {
            param.ty.span.lo()
        }
    } else {
        param.attrs[0].span.lo()
    }
}

pub(crate) fn span_hi_for_param(context: &RewriteContext<'_>, param: &ast::Param) -> BytePos {
    match param.ty.kind {
        ast::TyKind::Infer if context.snippet(param.ty.span) == "_" => param.ty.span.hi(),
        ast::TyKind::Infer if is_named_param(param) => param.pat.span.hi(),
        _ => param.ty.span.hi(),
    }
}

pub(crate) fn is_named_param(param: &ast::Param) -> bool {
    !matches!(param.pat.kind, ast::PatKind::Missing)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum FnBraceStyle {
    SameLine,
    NextLine,
    None,
}

// Return type is (result, force_new_line_for_brace)
fn rewrite_fn_base(
    context: &RewriteContext<'_>,
    indent: Indent,
    ident: symbol::Ident,
    fn_sig: &FnSig<'_>,
    span: Span,
    fn_brace_style: FnBraceStyle,
) -> Result<(String, bool, bool), RewriteError> {
    let mut force_new_line_for_brace = false;

    let where_clause = &fn_sig.generics.where_clause;

    let mut result = String::with_capacity(1024);
    result.push_str(&fn_sig.to_str(context));

    // fn foo
    result.push_str("fn ");

    // Generics.
    let overhead = if let FnBraceStyle::SameLine = fn_brace_style {
        // 4 = `() {`
        4
    } else {
        // 2 = `()`
        2
    };
    let used_width = last_line_used_width(&result, indent.width());
    let one_line_budget = context.budget(used_width + overhead);
    let shape = Shape {
        width: one_line_budget,
        indent,
        offset: used_width,
    };
    let fd = fn_sig.decl;
    let generics_str = rewrite_generics(
        context,
        rewrite_ident(context, ident),
        &fn_sig.generics,
        shape,
    )?;
    result.push_str(&generics_str);

    let snuggle_angle_bracket = generics_str
        .lines()
        .last()
        .map_or(false, |l| l.trim_start().len() == 1);

    // Note that the width and indent don't really matter, we'll re-layout the
    // return type later anyway.
    let ret_str = fd
        .output
        .rewrite_result(context, Shape::indented(indent, context.config))?;

    let multi_line_ret_str = ret_str.contains('\n');
    let ret_str_len = if multi_line_ret_str { 0 } else { ret_str.len() };

    // Params.
    let (one_line_budget, multi_line_budget, mut param_indent) = compute_budgets_for_params(
        context,
        &result,
        indent,
        ret_str_len,
        fn_brace_style,
        multi_line_ret_str,
    );

    debug!(
        "rewrite_fn_base: one_line_budget: {}, multi_line_budget: {}, param_indent: {:?}",
        one_line_budget, multi_line_budget, param_indent
    );

    result.push('(');
    // Check if vertical layout was forced.
    if one_line_budget == 0
        && !snuggle_angle_bracket
        && context.config.indent_style() == IndentStyle::Visual
    {
        result.push_str(&param_indent.to_string_with_newline(context.config));
    }

    let params_end = if fd.inputs.is_empty() {
        context
            .snippet_provider
            .span_after(mk_sp(fn_sig.generics.span.hi(), span.hi()), ")")
    } else {
        let last_span = mk_sp(fd.inputs[fd.inputs.len() - 1].span().hi(), span.hi());
        context.snippet_provider.span_after(last_span, ")")
    };
    let params_span = mk_sp(
        context
            .snippet_provider
            .span_after(mk_sp(fn_sig.generics.span.hi(), span.hi()), "("),
        params_end,
    );
    let param_str = rewrite_params(
        context,
        &fd.inputs,
        one_line_budget,
        multi_line_budget,
        indent,
        param_indent,
        params_span,
        fd.c_variadic(),
    )?;

    let put_params_in_block = match context.config.indent_style() {
        IndentStyle::Block => param_str.contains('\n') || param_str.len() > one_line_budget,
        _ => false,
    } && !fd.inputs.is_empty();

    let mut params_last_line_contains_comment = false;
    let mut no_params_and_over_max_width = false;

    if put_params_in_block {
        param_indent = indent.block_indent(context.config);
        result.push_str(&param_indent.to_string_with_newline(context.config));
        result.push_str(&param_str);
        result.push_str(&indent.to_string_with_newline(context.config));
        result.push(')');
    } else {
        result.push_str(&param_str);
        let used_width = last_line_used_width(&result, indent.width()) + first_line_width(&ret_str);
        // Put the closing brace on the next line if it overflows the max width.
        // 1 = `)`
        let closing_paren_overflow_max_width =
            fd.inputs.is_empty() && used_width + 1 > context.config.max_width();
        // If the last line of params contains comment, we cannot put the closing paren
        // on the same line.
        params_last_line_contains_comment = param_str
            .lines()
            .last()
            .map_or(false, |last_line| last_line.contains("//"));

        if context.config.style_edition() >= StyleEdition::Edition2024 {
            if closing_paren_overflow_max_width {
                result.push(')');
                result.push_str(&indent.to_string_with_newline(context.config));
                no_params_and_over_max_width = true;
            } else if params_last_line_contains_comment {
                result.push_str(&indent.to_string_with_newline(context.config));
                result.push(')');
                no_params_and_over_max_width = true;
            } else {
                result.push(')');
            }
        } else {
            if closing_paren_overflow_max_width || params_last_line_contains_comment {
                result.push_str(&indent.to_string_with_newline(context.config));
            }
            result.push(')');
        }
    }

    // Return type.
    if let ast::FnRetTy::Ty(..) = fd.output {
        let ret_should_indent = match context.config.indent_style() {
            // If our params are block layout then we surely must have space.
            IndentStyle::Block if put_params_in_block || fd.inputs.is_empty() => false,
            _ if params_last_line_contains_comment => false,
            _ if result.contains('\n') || multi_line_ret_str => true,
            _ => {
                // If the return type would push over the max width, then put the return type on
                // a new line. With the +1 for the signature length an additional space between
                // the closing parenthesis of the param and the arrow '->' is considered.
                let mut sig_length = result.len() + indent.width() + ret_str_len + 1;

                // If there is no where-clause, take into account the space after the return type
                // and the brace.
                if where_clause.predicates.is_empty() {
                    sig_length += 2;
                }

                sig_length > context.config.max_width()
            }
        };
        let ret_shape = if ret_should_indent {
            if context.config.style_edition() <= StyleEdition::Edition2021
                || context.config.indent_style() == IndentStyle::Visual
            {
                let indent = if param_str.is_empty() {
                    // Aligning with nonexistent params looks silly.
                    force_new_line_for_brace = true;
                    indent + 4
                } else {
                    // FIXME: we might want to check that using the param indent
                    // doesn't blow our budget, and if it does, then fallback to
                    // the where-clause indent.
                    param_indent
                };

                result.push_str(&indent.to_string_with_newline(context.config));
                Shape::indented(indent, context.config)
            } else {
                let mut ret_shape = Shape::indented(indent, context.config);
                if param_str.is_empty() {
                    // Aligning with nonexistent params looks silly.
                    force_new_line_for_brace = true;
                    ret_shape = if context.use_block_indent() {
                        ret_shape.offset_left(4).unwrap_or(ret_shape)
                    } else {
                        ret_shape.indent = ret_shape.indent + 4;
                        ret_shape
                    };
                }

                result.push_str(&ret_shape.indent.to_string_with_newline(context.config));
                ret_shape
            }
        } else {
            if context.config.style_edition() >= StyleEdition::Edition2024 {
                if !param_str.is_empty() || !no_params_and_over_max_width {
                    result.push(' ');
                }
            } else {
                result.push(' ');
            }

            let ret_shape = Shape::indented(indent, context.config);
            ret_shape
                .offset_left(last_line_width(&result))
                .unwrap_or(ret_shape)
        };

        if multi_line_ret_str || ret_should_indent {
            // Now that we know the proper indent and width, we need to
            // re-layout the return type.
            let ret_str = fd.output.rewrite_result(context, ret_shape)?;
            result.push_str(&ret_str);
        } else {
            result.push_str(&ret_str);
        }

        // Comment between return type and the end of the decl.
        let snippet_lo = fd.output.span().hi();
        if where_clause.predicates.is_empty() {
            let snippet_hi = span.hi();
            let snippet = context.snippet(mk_sp(snippet_lo, snippet_hi));
            // Try to preserve the layout of the original snippet.
            let original_starts_with_newline = snippet
                .find(|c| c != ' ')
                .map_or(false, |i| starts_with_newline(&snippet[i..]));
            let original_ends_with_newline = snippet
                .rfind(|c| c != ' ')
                .map_or(false, |i| snippet[i..].ends_with('\n'));
            let snippet = snippet.trim();
            if !snippet.is_empty() {
                result.push(if original_starts_with_newline {
                    '\n'
                } else {
                    ' '
                });
                result.push_str(snippet);
                if original_ends_with_newline {
                    force_new_line_for_brace = true;
                }
            }
        }
    }

    let pos_before_where = match fd.output {
        ast::FnRetTy::Default(..) => params_span.hi(),
        ast::FnRetTy::Ty(ref ty) => ty.span.hi(),
    };

    let is_params_multi_lined = param_str.contains('\n');

    let space = if put_params_in_block && ret_str.is_empty() {
        WhereClauseSpace::Space
    } else {
        WhereClauseSpace::Newline
    };
    let mut option = WhereClauseOption::new(fn_brace_style == FnBraceStyle::None, space);
    if is_params_multi_lined {
        option.veto_single_line();
    }
    let where_clause_str = rewrite_where_clause(
        context,
        &where_clause.predicates,
        where_clause.span,
        context.config.brace_style(),
        Shape::indented(indent, context.config),
        true,
        "{",
        Some(span.hi()),
        pos_before_where,
        option,
    )?;
    // If there are neither where-clause nor return type, we may be missing comments between
    // params and `{`.
    if where_clause_str.is_empty() {
        if let ast::FnRetTy::Default(ret_span) = fd.output {
            match recover_missing_comment_in_span(
                // from after the closing paren to right before block or semicolon
                mk_sp(ret_span.lo(), span.hi()),
                shape,
                context,
                last_line_width(&result),
            ) {
                Ok(ref missing_comment) if !missing_comment.is_empty() => {
                    result.push_str(missing_comment);
                    force_new_line_for_brace = true;
                }
                _ => (),
            }
        }
    }

    result.push_str(&where_clause_str);

    let ends_with_comment = last_line_contains_single_line_comment(&result);
    force_new_line_for_brace |= ends_with_comment;
    force_new_line_for_brace |=
        is_params_multi_lined && context.config.where_single_line() && !where_clause_str.is_empty();
    Ok((result, ends_with_comment, force_new_line_for_brace))
}

/// Kind of spaces to put before `where`.
#[derive(Copy, Clone)]
enum WhereClauseSpace {
    /// A single space.
    Space,
    /// A new line.
    Newline,
    /// Nothing.
    None,
}

#[derive(Copy, Clone)]
struct WhereClauseOption {
    suppress_comma: bool, // Force no trailing comma
    snuggle: WhereClauseSpace,
    allow_single_line: bool, // Try single line where-clause instead of vertical layout
    veto_single_line: bool,  // Disallow a single-line where-clause.
}

impl WhereClauseOption {
    fn new(suppress_comma: bool, snuggle: WhereClauseSpace) -> WhereClauseOption {
        WhereClauseOption {
            suppress_comma,
            snuggle,
            allow_single_line: false,
            veto_single_line: false,
        }
    }

    fn snuggled(current: &str) -> WhereClauseOption {
        WhereClauseOption {
            suppress_comma: false,
            snuggle: if last_line_width(current) == 1 {
                WhereClauseSpace::Space
            } else {
                WhereClauseSpace::Newline
            },
            allow_single_line: false,
            veto_single_line: false,
        }
    }

    fn suppress_comma(&mut self) {
        self.suppress_comma = true
    }

    fn allow_single_line(&mut self) {
        self.allow_single_line = true
    }

    fn snuggle(&mut self) {
        self.snuggle = WhereClauseSpace::Space
    }

    fn veto_single_line(&mut self) {
        self.veto_single_line = true;
    }
}

fn rewrite_params(
    context: &RewriteContext<'_>,
    params: &[ast::Param],
    one_line_budget: usize,
    multi_line_budget: usize,
    indent: Indent,
    param_indent: Indent,
    span: Span,
    variadic: bool,
) -> RewriteResult {
    if params.is_empty() {
        let comment = context
            .snippet(mk_sp(
                span.lo(),
                // to remove ')'
                span.hi() - BytePos(1),
            ))
            .trim();
        return Ok(comment.to_owned());
    }
    let param_items: Vec<_> = itemize_list(
        context.snippet_provider,
        params.iter(),
        ")",
        ",",
        |param| span_lo_for_param(param),
        |param| param.ty.span.hi(),
        |param| {
            param
                .rewrite_result(context, Shape::legacy(multi_line_budget, param_indent))
                .or_else(|_| Ok(context.snippet(param.span()).to_owned()))
        },
        span.lo(),
        span.hi(),
        false,
    )
    .collect();

    let tactic = definitive_tactic(
        &param_items,
        context
            .config
            .fn_params_layout()
            .to_list_tactic(param_items.len()),
        Separator::Comma,
        one_line_budget,
    );
    let budget = match tactic {
        DefinitiveListTactic::Horizontal => one_line_budget,
        _ => multi_line_budget,
    };
    let indent = match context.config.indent_style() {
        IndentStyle::Block => indent.block_indent(context.config),
        IndentStyle::Visual => param_indent,
    };
    let trailing_separator = if variadic {
        SeparatorTactic::Never
    } else {
        match context.config.indent_style() {
            IndentStyle::Block => context.config.trailing_comma(),
            IndentStyle::Visual => SeparatorTactic::Never,
        }
    };
    let fmt = ListFormatting::new(Shape::legacy(budget, indent), context.config)
        .tactic(tactic)
        .trailing_separator(trailing_separator)
        .ends_with_newline(tactic.ends_with_newline(context.config.indent_style()))
        .preserve_newline(true);
    write_list(&param_items, &fmt)
}

fn compute_budgets_for_params(
    context: &RewriteContext<'_>,
    result: &str,
    indent: Indent,
    ret_str_len: usize,
    fn_brace_style: FnBraceStyle,
    force_vertical_layout: bool,
) -> (usize, usize, Indent) {
    debug!(
        "compute_budgets_for_params {} {:?}, {}, {:?}",
        result.len(),
        indent,
        ret_str_len,
        fn_brace_style,
    );
    // Try keeping everything on the same line.
    if !result.contains('\n') && !force_vertical_layout {
        // 2 = `()`, 3 = `() `, space is before ret_string.
        let overhead = if ret_str_len == 0 { 2 } else { 3 };
        let mut used_space = indent.width() + result.len() + ret_str_len + overhead;
        match fn_brace_style {
            FnBraceStyle::None => used_space += 1,     // 1 = `;`
            FnBraceStyle::SameLine => used_space += 2, // 2 = `{}`
            FnBraceStyle::NextLine => (),
        }
        let one_line_budget = context.budget(used_space);

        if one_line_budget > 0 {
            // 4 = "() {".len()
            let (indent, multi_line_budget) = match context.config.indent_style() {
                IndentStyle::Block => {
                    let indent = indent.block_indent(context.config);
                    (indent, context.budget(indent.width() + 1))
                }
                IndentStyle::Visual => {
                    let indent = indent + result.len() + 1;
                    let multi_line_overhead = match fn_brace_style {
                        FnBraceStyle::SameLine => 4,
                        _ => 2,
                    } + indent.width();
                    (indent, context.budget(multi_line_overhead))
                }
            };

            return (one_line_budget, multi_line_budget, indent);
        }
    }

    // Didn't work. we must force vertical layout and put params on a newline.
    let new_indent = indent.block_indent(context.config);
    let used_space = match context.config.indent_style() {
        // 1 = `,`
        IndentStyle::Block => new_indent.width() + 1,
        // Account for `)` and possibly ` {`.
        IndentStyle::Visual => new_indent.width() + if ret_str_len == 0 { 1 } else { 3 },
    };
    (0, context.budget(used_space), new_indent)
}

fn newline_for_brace(config: &Config, where_clause: &ast::WhereClause) -> FnBraceStyle {
    let predicate_count = where_clause.predicates.len();

    if config.where_single_line() && predicate_count == 1 {
        return FnBraceStyle::SameLine;
    }
    let brace_style = config.brace_style();

    let use_next_line = brace_style == BraceStyle::AlwaysNextLine
        || (brace_style == BraceStyle::SameLineWhere && predicate_count > 0);
    if use_next_line {
        FnBraceStyle::NextLine
    } else {
        FnBraceStyle::SameLine
    }
}

fn rewrite_generics(
    context: &RewriteContext<'_>,
    ident: &str,
    generics: &ast::Generics,
    shape: Shape,
) -> RewriteResult {
    // FIXME: convert bounds to where-clauses where they get too big or if
    // there is a where-clause at all.

    if generics.params.is_empty() {
        return Ok(ident.to_owned());
    }

    let params = generics.params.iter();
    overflow::rewrite_with_angle_brackets(context, ident, params, shape, generics.span)
}

fn generics_shape_from_config(config: &Config, shape: Shape, offset: usize) -> Option<Shape> {
    match config.indent_style() {
        IndentStyle::Visual => shape.visual_indent(1 + offset).sub_width(offset + 2),
        IndentStyle::Block => {
            // 1 = ","
            shape
                .block()
                .block_indent(config.tab_spaces())
                .with_max_width(config)
                .sub_width(1)
        }
    }
}

fn rewrite_where_clause_rfc_style(
    context: &RewriteContext<'_>,
    predicates: &[ast::WherePredicate],
    where_span: Span,
    shape: Shape,
    terminator: &str,
    span_end: Option<BytePos>,
    span_end_before_where: BytePos,
    where_clause_option: WhereClauseOption,
) -> RewriteResult {
    let (where_keyword, allow_single_line) = rewrite_where_keyword(
        context,
        predicates,
        where_span,
        shape,
        span_end_before_where,
        where_clause_option,
    )?;

    // 1 = `,`
    let clause_shape = shape
        .block()
        .with_max_width(context.config)
        .block_left(context.config.tab_spaces())
        .and_then(|s| s.sub_width(1))
        .max_width_error(shape.width, where_span)?;
    let force_single_line = context.config.where_single_line()
        && predicates.len() == 1
        && !where_clause_option.veto_single_line;

    let preds_str = rewrite_bounds_on_where_clause(
        context,
        predicates,
        clause_shape,
        terminator,
        span_end,
        where_clause_option,
        force_single_line,
    )?;

    // 6 = `where `
    let clause_sep =
        if allow_single_line && !preds_str.contains('\n') && 6 + preds_str.len() <= shape.width
            || force_single_line
        {
            Cow::from(" ")
        } else {
            clause_shape.indent.to_string_with_newline(context.config)
        };

    Ok(format!("{where_keyword}{clause_sep}{preds_str}"))
}

/// Rewrite `where` and comment around it.
fn rewrite_where_keyword(
    context: &RewriteContext<'_>,
    predicates: &[ast::WherePredicate],
    where_span: Span,
    shape: Shape,
    span_end_before_where: BytePos,
    where_clause_option: WhereClauseOption,
) -> Result<(String, bool), RewriteError> {
    let block_shape = shape.block().with_max_width(context.config);
    // 1 = `,`
    let clause_shape = block_shape
        .block_left(context.config.tab_spaces())
        .and_then(|s| s.sub_width(1))
        .max_width_error(block_shape.width, where_span)?;

    let comment_separator = |comment: &str, shape: Shape| {
        if comment.is_empty() {
            Cow::from("")
        } else {
            shape.indent.to_string_with_newline(context.config)
        }
    };

    let (span_before, span_after) =
        missing_span_before_after_where(span_end_before_where, predicates, where_span);
    let (comment_before, comment_after) =
        rewrite_comments_before_after_where(context, span_before, span_after, shape)?;

    let starting_newline = match where_clause_option.snuggle {
        WhereClauseSpace::Space if comment_before.is_empty() => Cow::from(" "),
        WhereClauseSpace::None => Cow::from(""),
        _ => block_shape.indent.to_string_with_newline(context.config),
    };

    let newline_before_where = comment_separator(&comment_before, shape);
    let newline_after_where = comment_separator(&comment_after, clause_shape);
    let result = format!(
        "{starting_newline}{comment_before}{newline_before_where}where\
{newline_after_where}{comment_after}"
    );
    let allow_single_line = where_clause_option.allow_single_line
        && comment_before.is_empty()
        && comment_after.is_empty();

    Ok((result, allow_single_line))
}

/// Rewrite bounds on a where clause.
fn rewrite_bounds_on_where_clause(
    context: &RewriteContext<'_>,
    predicates: &[ast::WherePredicate],
    shape: Shape,
    terminator: &str,
    span_end: Option<BytePos>,
    where_clause_option: WhereClauseOption,
    force_single_line: bool,
) -> RewriteResult {
    let span_start = predicates[0].span().lo();
    // If we don't have the start of the next span, then use the end of the
    // predicates, but that means we miss comments.
    let len = predicates.len();
    let end_of_preds = predicates[len - 1].span().hi();
    let span_end = span_end.unwrap_or(end_of_preds);
    let items = itemize_list(
        context.snippet_provider,
        predicates.iter(),
        terminator,
        ",",
        |pred| pred.span().lo(),
        |pred| pred.span().hi(),
        |pred| pred.rewrite_result(context, shape),
        span_start,
        span_end,
        false,
    );
    let comma_tactic = if where_clause_option.suppress_comma || force_single_line {
        SeparatorTactic::Never
    } else {
        context.config.trailing_comma()
    };

    // shape should be vertical only and only if we have `force_single_line` option enabled
    // and the number of items of the where-clause is equal to 1
    let shape_tactic = if force_single_line {
        DefinitiveListTactic::Horizontal
    } else {
        DefinitiveListTactic::Vertical
    };

    let preserve_newline = context.config.style_edition() <= StyleEdition::Edition2021;

    let fmt = ListFormatting::new(shape, context.config)
        .tactic(shape_tactic)
        .trailing_separator(comma_tactic)
        .preserve_newline(preserve_newline);
    write_list(&items.collect::<Vec<_>>(), &fmt)
}

fn rewrite_where_clause(
    context: &RewriteContext<'_>,
    predicates: &[ast::WherePredicate],
    where_span: Span,
    brace_style: BraceStyle,
    shape: Shape,
    on_new_line: bool,
    terminator: &str,
    span_end: Option<BytePos>,
    span_end_before_where: BytePos,
    where_clause_option: WhereClauseOption,
) -> RewriteResult {
    if predicates.is_empty() {
        return Ok(String::new());
    }

    if context.config.indent_style() == IndentStyle::Block {
        return rewrite_where_clause_rfc_style(
            context,
            predicates,
            where_span,
            shape,
            terminator,
            span_end,
            span_end_before_where,
            where_clause_option,
        );
    }

    let extra_indent = Indent::new(context.config.tab_spaces(), 0);

    let offset = match context.config.indent_style() {
        IndentStyle::Block => shape.indent + extra_indent.block_indent(context.config),
        // 6 = "where ".len()
        IndentStyle::Visual => shape.indent + extra_indent + 6,
    };
    // FIXME: if indent_style != Visual, then the budgets below might
    // be out by a char or two.

    let budget = context.config.max_width() - offset.width();
    let span_start = predicates[0].span().lo();
    // If we don't have the start of the next span, then use the end of the
    // predicates, but that means we miss comments.
    let len = predicates.len();
    let end_of_preds = predicates[len - 1].span().hi();
    let span_end = span_end.unwrap_or(end_of_preds);
    let items = itemize_list(
        context.snippet_provider,
        predicates.iter(),
        terminator,
        ",",
        |pred| pred.span().lo(),
        |pred| pred.span().hi(),
        |pred| pred.rewrite_result(context, Shape::legacy(budget, offset)),
        span_start,
        span_end,
        false,
    );
    let item_vec = items.collect::<Vec<_>>();
    // FIXME: we don't need to collect here
    let tactic = definitive_tactic(&item_vec, ListTactic::Vertical, Separator::Comma, budget);

    let mut comma_tactic = context.config.trailing_comma();
    // Kind of a hack because we don't usually have trailing commas in where-clauses.
    if comma_tactic == SeparatorTactic::Vertical || where_clause_option.suppress_comma {
        comma_tactic = SeparatorTactic::Never;
    }

    let fmt = ListFormatting::new(Shape::legacy(budget, offset), context.config)
        .tactic(tactic)
        .trailing_separator(comma_tactic)
        .ends_with_newline(tactic.ends_with_newline(context.config.indent_style()))
        .preserve_newline(true);
    let preds_str = write_list(&item_vec, &fmt)?;

    let end_length = if terminator == "{" {
        // If the brace is on the next line we don't need to count it otherwise it needs two
        // characters " {"
        match brace_style {
            BraceStyle::AlwaysNextLine | BraceStyle::SameLineWhere => 0,
            BraceStyle::PreferSameLine => 2,
        }
    } else if terminator == "=" {
        2
    } else {
        terminator.len()
    };
    if on_new_line
        || preds_str.contains('\n')
        || shape.indent.width() + " where ".len() + preds_str.len() + end_length > shape.width
    {
        Ok(format!(
            "\n{}where {}",
            (shape.indent + extra_indent).to_string(context.config),
            preds_str
        ))
    } else {
        Ok(format!(" where {preds_str}"))
    }
}

fn missing_span_before_after_where(
    before_item_span_end: BytePos,
    predicates: &[ast::WherePredicate],
    where_span: Span,
) -> (Span, Span) {
    let missing_span_before = mk_sp(before_item_span_end, where_span.lo());
    // 5 = `where`
    let pos_after_where = where_span.lo() + BytePos(5);
    let missing_span_after = mk_sp(pos_after_where, predicates[0].span().lo());
    (missing_span_before, missing_span_after)
}

fn rewrite_comments_before_after_where(
    context: &RewriteContext<'_>,
    span_before_where: Span,
    span_after_where: Span,
    shape: Shape,
) -> Result<(String, String), RewriteError> {
    let before_comment = rewrite_missing_comment(span_before_where, shape, context)?;
    let after_comment = rewrite_missing_comment(
        span_after_where,
        shape.block_indent(context.config.tab_spaces()),
        context,
    )?;
    Ok((before_comment, after_comment))
}

fn format_header(
    context: &RewriteContext<'_>,
    item_name: &str,
    ident: symbol::Ident,
    vis: &ast::Visibility,
    offset: Indent,
) -> String {
    let mut result = String::with_capacity(128);
    let shape = Shape::indented(offset, context.config);

    result.push_str(format_visibility(context, vis).trim());

    // Check for a missing comment between the visibility and the item name.
    let after_vis = vis.span.hi();
    if let Some(before_item_name) = context
        .snippet_provider
        .opt_span_before(mk_sp(vis.span.lo(), ident.span.hi()), item_name.trim())
    {
        let missing_span = mk_sp(after_vis, before_item_name);
        if let Ok(result_with_comment) = combine_strs_with_missing_comments(
            context,
            &result,
            item_name,
            missing_span,
            shape,
            /* allow_extend */ true,
        ) {
            result = result_with_comment;
        }
    }

    result.push_str(rewrite_ident(context, ident));

    result
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum BracePos {
    None,
    Auto,
    ForceSameLine,
}

fn format_generics(
    context: &RewriteContext<'_>,
    generics: &ast::Generics,
    brace_style: BraceStyle,
    brace_pos: BracePos,
    offset: Indent,
    span: Span,
    used_width: usize,
) -> Option<String> {
    let shape = Shape::legacy(context.budget(used_width + offset.width()), offset);
    let mut result = rewrite_generics(context, "", generics, shape).ok()?;

    // If the generics are not parameterized then generics.span.hi() == 0,
    // so we use span.lo(), which is the position after `struct Foo`.
    let span_end_before_where = if !generics.params.is_empty() {
        generics.span.hi()
    } else {
        span.lo()
    };
    let (same_line_brace, missed_comments) = if !generics.where_clause.predicates.is_empty() {
        let budget = context.budget(last_line_used_width(&result, offset.width()));
        let mut option = WhereClauseOption::snuggled(&result);
        if brace_pos == BracePos::None {
            option.suppress_comma = true;
        }
        let where_clause_str = rewrite_where_clause(
            context,
            &generics.where_clause.predicates,
            generics.where_clause.span,
            brace_style,
            Shape::legacy(budget, offset.block_only()),
            true,
            "{",
            Some(span.hi()),
            span_end_before_where,
            option,
        )
        .ok()?;
        result.push_str(&where_clause_str);
        (
            brace_pos == BracePos::ForceSameLine || brace_style == BraceStyle::PreferSameLine,
            // missed comments are taken care of in #rewrite_where_clause
            None,
        )
    } else {
        (
            brace_pos == BracePos::ForceSameLine
                || (result.contains('\n') && brace_style == BraceStyle::PreferSameLine
                    || brace_style != BraceStyle::AlwaysNextLine)
                || trimmed_last_line_width(&result) == 1,
            rewrite_missing_comment(
                mk_sp(
                    span_end_before_where,
                    if brace_pos == BracePos::None {
                        span.hi()
                    } else {
                        context.snippet_provider.span_before_last(span, "{")
                    },
                ),
                shape,
                context,
            )
            .ok(),
        )
    };
    // add missing comments
    let missed_line_comments = missed_comments
        .filter(|missed_comments| !missed_comments.is_empty())
        .map_or(false, |missed_comments| {
            let is_block = is_last_comment_block(&missed_comments);
            let sep = if is_block { " " } else { "\n" };
            result.push_str(sep);
            result.push_str(&missed_comments);
            !is_block
        });
    if brace_pos == BracePos::None {
        return Some(result);
    }
    let total_used_width = last_line_used_width(&result, used_width);
    let remaining_budget = context.budget(total_used_width);
    // If the same line brace if forced, it indicates that we are rewriting an item with empty body,
    // and hence we take the closer into account as well for one line budget.
    // We assume that the closer has the same length as the opener.
    let overhead = if brace_pos == BracePos::ForceSameLine {
        // 3 = ` {}`
        3
    } else {
        // 2 = ` {`
        2
    };
    let forbid_same_line_brace = missed_line_comments || overhead > remaining_budget;
    if !forbid_same_line_brace && same_line_brace {
        result.push(' ');
    } else {
        result.push('\n');
        result.push_str(&offset.block_only().to_string(context.config));
    }
    result.push('{');

    Some(result)
}

impl Rewrite for ast::ForeignItem {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        self.rewrite_result(context, shape).ok()
    }

    fn rewrite_result(&self, context: &RewriteContext<'_>, shape: Shape) -> RewriteResult {
        let attrs_str = self.attrs.rewrite_result(context, shape)?;
        // Drop semicolon or it will be interpreted as comment.
        // FIXME: this may be a faulty span from libsyntax.
        let span = mk_sp(self.span.lo(), self.span.hi() - BytePos(1));

        let item_str = match self.kind {
            ast::ForeignItemKind::Fn(ref fn_kind) => {
                let ast::Fn {
                    defaultness,
                    ref sig,
                    ident,
                    ref generics,
                    ref body,
                    ..
                } = **fn_kind;
                if body.is_some() {
                    let mut visitor = FmtVisitor::from_context(context);
                    visitor.block_indent = shape.indent;
                    visitor.last_pos = self.span.lo();
                    let inner_attrs = inner_attributes(&self.attrs);
                    let fn_ctxt = visit::FnCtxt::Foreign;
                    visitor.visit_fn(
                        ident,
                        visit::FnKind::Fn(fn_ctxt, &self.vis, fn_kind),
                        &sig.decl,
                        self.span,
                        defaultness,
                        Some(&inner_attrs),
                    );
                    Ok(visitor.buffer.to_owned())
                } else {
                    rewrite_fn_base(
                        context,
                        shape.indent,
                        ident,
                        &FnSig::from_method_sig(sig, generics, &self.vis),
                        span,
                        FnBraceStyle::None,
                    )
                    .map(|(s, _, _)| format!("{};", s))
                }
            }
            ast::ForeignItemKind::Static(ref static_foreign_item) => {
                // FIXME(#21): we're dropping potential comments in between the
                // function kw here.
                let vis = format_visibility(context, &self.vis);
                let safety = format_safety(static_foreign_item.safety);
                let mut_str = format_mutability(static_foreign_item.mutability);
                let prefix = format!(
                    "{}{}static {}{}:",
                    vis,
                    safety,
                    mut_str,
                    rewrite_ident(context, static_foreign_item.ident)
                );
                // 1 = ;
                rewrite_assign_rhs(
                    context,
                    prefix,
                    &static_foreign_item.ty,
                    &RhsAssignKind::Ty,
                    shape
                        .sub_width(1)
                        .max_width_error(shape.width, static_foreign_item.ty.span)?,
                )
                .map(|s| s + ";")
            }
            ast::ForeignItemKind::TyAlias(ref ty_alias) => {
                let kind = ItemVisitorKind::ForeignItem;
                rewrite_type_alias(ty_alias, &self.vis, context, shape.indent, kind, self.span)
            }
            ast::ForeignItemKind::MacCall(ref mac) => {
                rewrite_macro(mac, context, shape, MacroPosition::Item)
            }
        }?;

        let missing_span = if self.attrs.is_empty() {
            mk_sp(self.span.lo(), self.span.lo())
        } else {
            mk_sp(self.attrs[self.attrs.len() - 1].span.hi(), self.span.lo())
        };
        combine_strs_with_missing_comments(
            context,
            &attrs_str,
            &item_str,
            missing_span,
            shape,
            false,
        )
    }
}

/// Rewrite the attributes of an item.
fn rewrite_attrs(
    context: &RewriteContext<'_>,
    item: &ast::Item,
    item_str: &str,
    shape: Shape,
) -> Option<String> {
    let attrs = filter_inline_attrs(&item.attrs, item.span());
    let attrs_str = attrs.rewrite(context, shape)?;

    let missed_span = if attrs.is_empty() {
        mk_sp(item.span.lo(), item.span.lo())
    } else {
        mk_sp(attrs[attrs.len() - 1].span.hi(), item.span.lo())
    };

    let allow_extend = if attrs.len() == 1 {
        let line_len = attrs_str.len() + 1 + item_str.len();
        !attrs.first().unwrap().is_doc_comment()
            && context.config.inline_attribute_width() >= line_len
    } else {
        false
    };

    combine_strs_with_missing_comments(
        context,
        &attrs_str,
        item_str,
        missed_span,
        shape,
        allow_extend,
    )
    .ok()
}

/// Rewrite an inline mod.
/// The given shape is used to format the mod's attributes.
pub(crate) fn rewrite_mod(
    context: &RewriteContext<'_>,
    item: &ast::Item,
    ident: Ident,
    attrs_shape: Shape,
) -> Option<String> {
    let mut result = String::with_capacity(32);
    result.push_str(&*format_visibility(context, &item.vis));
    result.push_str("mod ");
    result.push_str(rewrite_ident(context, ident));
    result.push(';');
    rewrite_attrs(context, item, &result, attrs_shape)
}

/// Rewrite `extern crate foo;`.
/// The given shape is used to format the extern crate's attributes.
pub(crate) fn rewrite_extern_crate(
    context: &RewriteContext<'_>,
    item: &ast::Item,
    attrs_shape: Shape,
) -> Option<String> {
    assert!(is_extern_crate(item));
    let new_str = context.snippet(item.span);
    let item_str = if contains_comment(new_str) {
        new_str.to_owned()
    } else {
        let no_whitespace = &new_str.split_whitespace().collect::<Vec<&str>>().join(" ");
        String::from(&*Regex::new(r"\s;").unwrap().replace(no_whitespace, ";"))
    };
    rewrite_attrs(context, item, &item_str, attrs_shape)
}

/// Returns `true` for `mod foo;`, false for `mod foo { .. }`.
pub(crate) fn is_mod_decl(item: &ast::Item) -> bool {
    !matches!(
        item.kind,
        ast::ItemKind::Mod(_, _, ast::ModKind::Loaded(_, ast::Inline::Yes, _))
    )
}

pub(crate) fn is_use_item(item: &ast::Item) -> bool {
    matches!(item.kind, ast::ItemKind::Use(_))
}

pub(crate) fn is_extern_crate(item: &ast::Item) -> bool {
    matches!(item.kind, ast::ItemKind::ExternCrate(..))
}
