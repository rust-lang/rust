// Formatting top-level items - functions, structs, enums, traits, impls.

use std::borrow::Cow;
use std::cmp::{max, min, Ordering};

use regex::Regex;
use rustc_ast::visit;
use rustc_ast::{ast, ptr};
use rustc_span::{symbol, BytePos, Span, DUMMY_SP};

use crate::attr::filter_inline_attrs;
use crate::comment::{
    combine_strs_with_missing_comments, contains_comment, is_last_comment_block,
    recover_comment_removed, recover_missing_comment_in_span, rewrite_missing_comment,
    FindUncommented,
};
use crate::config::lists::*;
use crate::config::{BraceStyle, Config, IndentStyle, Version};
use crate::expr::{
    is_empty_block, is_simple_block_stmt, rewrite_assign_rhs, rewrite_assign_rhs_with,
    rewrite_assign_rhs_with_comments, RhsTactics,
};
use crate::lists::{definitive_tactic, itemize_list, write_list, ListFormatting, Separator};
use crate::macros::{rewrite_macro, MacroPosition};
use crate::overflow;
use crate::rewrite::{Rewrite, RewriteContext};
use crate::shape::{Indent, Shape};
use crate::source_map::{LineRangeUtils, SpanUtils};
use crate::spanned::Spanned;
use crate::stmt::Stmt;
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
// let pat: ty = init;
impl Rewrite for ast::Local {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        debug!(
            "Local::rewrite {:?} {} {:?}",
            self, shape.width, shape.indent
        );

        skip_out_of_file_lines_range!(context, self.span);

        if contains_skip(&self.attrs) || matches!(self.kind, ast::LocalKind::InitElse(..)) {
            return None;
        }

        let attrs_str = self.attrs.rewrite(context, shape)?;
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

        // 4 = "let ".len()
        let pat_shape = shape.offset_left(4)?;
        // 1 = ;
        let pat_shape = pat_shape.sub_width(1)?;
        let pat_str = self.pat.rewrite(context, pat_shape)?;
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
                .offset_left(last_line_width(&result) + separator.len())?
                // 2 = ` =`
                .sub_width(2)?;

                let rewrite = ty.rewrite(context, ty_shape)?;

                infix.push_str(separator);
                infix.push_str(&rewrite);
            }

            if self.kind.init().is_some() {
                infix.push_str(" =");
            }

            infix
        };

        result.push_str(&infix);

        if let Some((init, _els)) = self.kind.init_else_opt() {
            // 1 = trailing semicolon;
            let nested_shape = shape.sub_width(1)?;

            result = rewrite_assign_rhs(context, result, init, nested_shape)?;
            // todo else
        }

        result.push(';');
        Some(result)
    }
}

// FIXME convert to using rewrite style rather than visitor
// FIXME format modules in this style
#[allow(dead_code)]
#[derive(Debug)]
struct Item<'a> {
    unsafety: ast::Unsafe,
    abi: Cow<'static, str>,
    vis: Option<&'a ast::Visibility>,
    body: Vec<BodyElement<'a>>,
    span: Span,
}

impl<'a> Item<'a> {
    fn from_foreign_mod(fm: &'a ast::ForeignMod, span: Span, config: &Config) -> Item<'a> {
        Item {
            unsafety: fm.unsafety,
            abi: format_extern(
                ast::Extern::from_abi(fm.abi),
                config.force_explicit_abi(),
                true,
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
    is_async: Cow<'a, ast::Async>,
    constness: ast::Const,
    defaultness: ast::Defaultness,
    unsafety: ast::Unsafe,
    visibility: &'a ast::Visibility,
}

impl<'a> FnSig<'a> {
    pub(crate) fn from_method_sig(
        method_sig: &'a ast::FnSig,
        generics: &'a ast::Generics,
        visibility: &'a ast::Visibility,
    ) -> FnSig<'a> {
        FnSig {
            unsafety: method_sig.header.unsafety,
            is_async: Cow::Borrowed(&method_sig.header.asyncness),
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
        generics: &'a ast::Generics,
        decl: &'a ast::FnDecl,
        defaultness: ast::Defaultness,
    ) -> FnSig<'a> {
        match *fn_kind {
            visit::FnKind::Fn(fn_ctxt, _, fn_sig, vis, _) => match fn_ctxt {
                visit::FnCtxt::Assoc(..) => {
                    let mut fn_sig = FnSig::from_method_sig(fn_sig, generics, vis);
                    fn_sig.defaultness = defaultness;
                    fn_sig
                }
                _ => FnSig {
                    decl,
                    generics,
                    ext: fn_sig.header.ext,
                    constness: fn_sig.header.constness,
                    is_async: Cow::Borrowed(&fn_sig.header.asyncness),
                    defaultness,
                    unsafety: fn_sig.header.unsafety,
                    visibility: vis,
                },
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
        result.push_str(format_async(&self.is_async));
        result.push_str(format_unsafety(self.unsafety));
        result.push_str(&format_extern(
            self.ext,
            context.config.force_explicit_abi(),
            false,
        ));
        result
    }
}

impl<'a> FmtVisitor<'a> {
    fn format_item(&mut self, item: &Item<'_>) {
        self.buffer.push_str(format_unsafety(item.unsafety));
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
            rewrite_fn_base(&context, indent, ident, fn_sig, span, fn_brace_style)?;

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
    ) -> Option<String> {
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

        Some(result)
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
            return Some(format!("{} {{}}", fn_str));
        }

        if !self.config.fn_single_line() || !is_simple_block_stmt(&context, block, None) {
            return None;
        }

        let res = Stmt::from_ast_node(block.stmts.first()?, true)
            .rewrite(&self.get_context(), self.shape())?;

        let width = self.block_indent.width() + fn_str.len() + res.len() + 5;
        if !res.contains('\n') && width <= self.config.max_width() {
            Some(format!("{} {{ {} }}", fn_str, res))
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
                |f| self.format_variant(f, one_line_width, pad_discrim_ident_to),
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

        let list = write_list(&items, &fmt)?;
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
        // 1 = ','
        let shape = self.shape().sub_width(1)?;
        let attrs_str = field.attrs.rewrite(&context, shape)?;
        let lo = field
            .attrs
            .last()
            .map_or(field.span.lo(), |attr| attr.span.hi());
        let span = mk_sp(lo, field.span.lo());

        let variant_body = match field.data {
            ast::VariantData::Tuple(..) | ast::VariantData::Struct(..) => format_struct(
                &context,
                &StructParts::from_variant(field),
                self.block_indent,
                Some(one_line_width),
            )?,
            ast::VariantData::Unit(..) => rewrite_ident(&context, field.ident).to_owned(),
        };

        let variant_body = if let Some(ref expr) = field.disr_expr {
            let lhs = format!("{:1$} =", variant_body, pad_discrim_ident_to);
            rewrite_assign_rhs_with(
                &context,
                lhs,
                &*expr.value,
                shape,
                RhsTactics::AllowOverflow,
            )?
        } else {
            variant_body
        };

        combine_strs_with_missing_comments(&context, &attrs_str, &variant_body, span, shape, false)
    }

    fn visit_impl_items(&mut self, items: &[ptr::P<ast::AssocItem>]) {
        if self.get_context().config.reorder_impl_items() {
            // Create visitor for each items, then reorder them.
            let mut buffer = vec![];
            for item in items {
                self.visit_impl_item(item);
                buffer.push((self.buffer.clone(), item.clone()));
                self.buffer.clear();
            }

            fn is_type(ty: &Option<rustc_ast::ptr::P<ast::Ty>>) -> bool {
                if let Some(lty) = ty {
                    if let ast::TyKind::ImplTrait(..) = lty.kind {
                        return false;
                    }
                }
                true
            }

            fn is_opaque(ty: &Option<rustc_ast::ptr::P<ast::Ty>>) -> bool {
                !is_type(ty)
            }

            fn both_type(
                a: &Option<rustc_ast::ptr::P<ast::Ty>>,
                b: &Option<rustc_ast::ptr::P<ast::Ty>>,
            ) -> bool {
                is_type(a) && is_type(b)
            }

            fn both_opaque(
                a: &Option<rustc_ast::ptr::P<ast::Ty>>,
                b: &Option<rustc_ast::ptr::P<ast::Ty>>,
            ) -> bool {
                is_opaque(a) && is_opaque(b)
            }

            // In rustc-ap-v638 the `OpaqueTy` AssocItemKind variant was removed but
            // we still need to differentiate to maintain sorting order.

            // type -> opaque -> const -> macro -> method
            use crate::ast::AssocItemKind::*;
            fn need_empty_line(a: &ast::AssocItemKind, b: &ast::AssocItemKind) -> bool {
                match (a, b) {
                    (TyAlias(lty), TyAlias(rty))
                        if both_type(&lty.ty, &rty.ty) || both_opaque(&lty.ty, &rty.ty) =>
                    {
                        false
                    }
                    (Const(..), Const(..)) => false,
                    _ => true,
                }
            }

            buffer.sort_by(|(_, a), (_, b)| match (&a.kind, &b.kind) {
                (TyAlias(lty), TyAlias(rty))
                    if both_type(&lty.ty, &rty.ty) || both_opaque(&lty.ty, &rty.ty) =>
                {
                    a.ident.as_str().cmp(&b.ident.as_str())
                }
                (Const(..), Const(..)) | (MacCall(..), MacCall(..)) => {
                    a.ident.as_str().cmp(&b.ident.as_str())
                }
                (Fn(..), Fn(..)) => a.span.lo().cmp(&b.span.lo()),
                (TyAlias(ty), _) if is_type(&ty.ty) => Ordering::Less,
                (_, TyAlias(ty)) if is_type(&ty.ty) => Ordering::Greater,
                (TyAlias(..), _) => Ordering::Less,
                (_, TyAlias(..)) => Ordering::Greater,
                (Const(..), _) => Ordering::Less,
                (_, Const(..)) => Ordering::Greater,
                (MacCall(..), _) => Ordering::Less,
                (_, MacCall(..)) => Ordering::Greater,
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
    offset: Indent,
) -> Option<String> {
    if let ast::ItemKind::Impl(impl_kind) = &item.kind {
        let ast::Impl {
            ref generics,
            ref self_ty,
            ref items,
            ..
        } = **impl_kind;
        let mut result = String::with_capacity(128);
        let ref_and_type = format_impl_ref_and_type(context, item, offset)?;
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
            &generics.where_clause,
            context.config.brace_style(),
            Shape::legacy(where_budget, offset.block_only()),
            false,
            "{",
            where_span_end,
            self_ty.span.hi(),
            option,
        )?;

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
                    Some(ref missing_comment) if !missing_comment.is_empty() => {
                        result.push_str(missing_comment);
                    }
                    _ => (),
                }
            }
        }

        if is_impl_single_line(context, items.as_slice(), &result, &where_clause_str, item)? {
            result.push_str(&where_clause_str);
            if where_clause_str.contains('\n') || last_line_contains_single_line_comment(&result) {
                // if the where_clause contains extra comments AND
                // there is only one where-clause predicate
                // recover the suppressed comma in single line where_clause formatting
                if generics.where_clause.predicates.len() == 1 {
                    result.push(',');
                }
                result.push_str(&format!("{}{{{}}}", sep, sep));
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
    } else {
        unreachable!();
    }
}

fn is_impl_single_line(
    context: &RewriteContext<'_>,
    items: &[ptr::P<ast::AssocItem>],
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
    offset: Indent,
) -> Option<String> {
    if let ast::ItemKind::Impl(impl_kind) = &item.kind {
        let ast::Impl {
            unsafety,
            polarity,
            defaultness,
            constness,
            ref generics,
            of_trait: ref trait_ref,
            ref self_ty,
            ..
        } = **impl_kind;
        let mut result = String::with_capacity(128);

        result.push_str(&format_visibility(context, &item.vis));
        result.push_str(format_defaultness(defaultness));
        result.push_str(format_unsafety(unsafety));

        let shape = if context.config.version() == Version::Two {
            Shape::indented(offset + last_line_width(&result), context.config)
        } else {
            generics_shape_from_config(
                context.config,
                Shape::indented(offset + last_line_width(&result), context.config),
                0,
            )?
        };
        let generics_str = rewrite_generics(context, "impl", generics, shape)?;
        result.push_str(&generics_str);
        result.push_str(format_constness_right(constness));

        let polarity_str = match polarity {
            ast::ImplPolarity::Negative(_) => "!",
            ast::ImplPolarity::Positive => "",
        };

        let polarity_overhead;
        let trait_ref_overhead;
        if let Some(ref trait_ref) = *trait_ref {
            let result_len = last_line_width(&result);
            result.push_str(&rewrite_trait_ref(
                context,
                trait_ref,
                offset,
                polarity_str,
                result_len,
            )?);
            polarity_overhead = 0; // already written
            trait_ref_overhead = " for".len();
        } else {
            polarity_overhead = polarity_str.len();
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
        let used_space = last_line_width(&result)
            + polarity_overhead
            + trait_ref_overhead
            + curly_brace_overhead;
        // 1 = space before the type.
        let budget = context.budget(used_space + 1);
        if let Some(self_ty_str) = self_ty.rewrite(context, Shape::legacy(budget, offset)) {
            if !self_ty_str.contains('\n') {
                if trait_ref.is_some() {
                    result.push_str(" for ");
                } else {
                    result.push(' ');
                    result.push_str(polarity_str);
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
        if trait_ref.is_some() {
            result.push_str("for ");
        } else {
            result.push_str(polarity_str);
        }
        let budget = context.budget(last_line_width(&result) + polarity_overhead);
        let type_offset = match context.config.indent_style() {
            IndentStyle::Visual => new_line_offset + trait_ref_overhead,
            IndentStyle::Block => new_line_offset,
        };
        result.push_str(&*self_ty.rewrite(context, Shape::legacy(budget, type_offset))?);
        Some(result)
    } else {
        unreachable!();
    }
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
            return Some(format!(" {}{}", polarity_str, trait_ref_str));
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

    fn from_variant(variant: &'a ast::Variant) -> Self {
        StructParts {
            prefix: "",
            ident: variant.ident,
            vis: &DEFAULT_VISIBILITY,
            def: &variant.data,
            generics: None,
            span: variant.span,
        }
    }

    pub(crate) fn from_item(item: &'a ast::Item) -> Self {
        let (prefix, def, generics) = match item.kind {
            ast::ItemKind::Struct(ref def, ref generics) => ("struct ", def, generics),
            ast::ItemKind::Union(ref def, ref generics) => ("union ", def, generics),
            _ => unreachable!(),
        };
        StructParts {
            prefix,
            ident: item.ident,
            vis: &item.vis,
            def,
            generics: Some(generics),
            span: item.span,
        }
    }
}

fn format_struct(
    context: &RewriteContext<'_>,
    struct_parts: &StructParts<'_>,
    offset: Indent,
    one_line_width: Option<usize>,
) -> Option<String> {
    match *struct_parts.def {
        ast::VariantData::Unit(..) => format_unit_struct(context, struct_parts, offset),
        ast::VariantData::Tuple(ref fields, _) => {
            format_tuple_struct(context, struct_parts, fields, offset)
        }
        ast::VariantData::Struct(ref fields, _) => {
            format_struct_struct(context, struct_parts, fields, offset, one_line_width)
        }
    }
}

pub(crate) fn format_trait(
    context: &RewriteContext<'_>,
    item: &ast::Item,
    offset: Indent,
) -> Option<String> {
    if let ast::ItemKind::Trait(trait_kind) = &item.kind {
        let ast::Trait {
            is_auto,
            unsafety,
            ref generics,
            ref bounds,
            ref items,
        } = **trait_kind;
        let mut result = String::with_capacity(128);
        let header = format!(
            "{}{}{}trait ",
            format_visibility(context, &item.vis),
            format_unsafety(unsafety),
            format_auto(is_auto),
        );
        result.push_str(&header);

        let body_lo = context.snippet_provider.span_after(item.span, "{");

        let shape = Shape::indented(offset, context.config).offset_left(result.len())?;
        let generics_str =
            rewrite_generics(context, rewrite_ident(context, item.ident), generics, shape)?;
        result.push_str(&generics_str);

        // FIXME(#2055): rustfmt fails to format when there are comments between trait bounds.
        if !bounds.is_empty() {
            let ident_hi = context
                .snippet_provider
                .span_after(item.span, &item.ident.as_str());
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
                RhsTactics::ForceNextLineWithoutIndent,
            )?;
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
                &generics.where_clause,
                context.config.brace_style(),
                Shape::legacy(where_budget, offset.block_only()),
                where_on_new_line,
                "{",
                None,
                pos_before_where,
                option,
            )?;
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
                let comment_hi = body_lo - BytePos(1);
                let comment_lo = item.span.lo() + BytePos(lo as u32);
                if comment_lo < comment_hi {
                    match recover_missing_comment_in_span(
                        mk_sp(comment_lo, comment_hi),
                        Shape::indented(offset, context.config),
                        context,
                        last_line_width(&result),
                    ) {
                        Some(ref missing_comment) if !missing_comment.is_empty() => {
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
    } else {
        unreachable!();
    }
}

pub(crate) struct TraitAliasBounds<'a> {
    generic_bounds: &'a ast::GenericBounds,
    generics: &'a ast::Generics,
}

impl<'a> Rewrite for TraitAliasBounds<'a> {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        let generic_bounds_str = self.generic_bounds.rewrite(context, shape)?;

        let mut option = WhereClauseOption::new(true, WhereClauseSpace::None);
        option.allow_single_line();

        let where_str = rewrite_where_clause(
            context,
            &self.generics.where_clause,
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

        Some(format!("{}{}{}", generic_bounds_str, space, where_str))
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
    let generics_str = rewrite_generics(context, alias, generics, g_shape)?;
    let vis_str = format_visibility(context, vis);
    let lhs = format!("{}trait {} =", vis_str, generics_str);
    // 1 = ";"
    let trait_alias_bounds = TraitAliasBounds {
        generic_bounds,
        generics,
    };
    rewrite_assign_rhs(context, lhs, &trait_alias_bounds, shape.sub_width(1)?).map(|s| s + ";")
}

fn format_unit_struct(
    context: &RewriteContext<'_>,
    p: &StructParts<'_>,
    offset: Indent,
) -> Option<String> {
    let header_str = format_header(context, p.prefix, p.ident, p.vis, offset);
    let generics_str = if let Some(generics) = p.generics {
        let hi = context.snippet_provider.span_before(p.span, ";");
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
    Some(format!("{}{};", header_str, generics_str))
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
    let body_lo = context.snippet_provider.span_after(span, "{");

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
        Some(format!("{} {} }}", result, items_str))
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
        ast::VisibilityKind::Crate(..) | ast::VisibilityKind::Restricted { .. } => vis.span.hi(),
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
    match rewrite_missing_comment(span, Shape::indented(offset, context.config), context) {
        Some(ref s) if s.is_empty() => (),
        Some(ref s) => {
            if !is_single_line(s) || first_line_contains_single_line_comment(s) {
                let nested_indent_str = offset
                    .block_indent(context.config)
                    .to_string_with_newline(context.config);
                result.push_str(&nested_indent_str);
            }
            result.push_str(s);
            if last_line_contains_single_line_comment(s) {
                result.push_str(&offset.to_string_with_newline(context.config));
            }
        }
        None => result.push_str(context.snippet(span)),
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
            let generics_str = rewrite_generics(context, "", generics, shape)?;
            result.push_str(&generics_str);

            let where_budget = context.budget(last_line_width(&result));
            let option = WhereClauseOption::new(true, WhereClauseSpace::Newline);
            rewrite_where_clause(
                context,
                &generics.where_clause,
                context.config.brace_style(),
                Shape::legacy(where_budget, offset.block_only()),
                false,
                ";",
                None,
                body_hi,
                option,
            )?
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
        )?;
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

pub(crate) enum ItemVisitorKind<'a> {
    Item(&'a ast::Item),
    AssocTraitItem(&'a ast::AssocItem),
    AssocImplItem(&'a ast::AssocItem),
    ForeignItem(&'a ast::ForeignItem),
}

struct TyAliasRewriteInfo<'c, 'g>(
    &'c RewriteContext<'c>,
    Indent,
    &'g ast::Generics,
    symbol::Ident,
    Span,
);

pub(crate) fn rewrite_type_alias<'a, 'b>(
    ty_alias_kind: &ast::TyAlias,
    context: &RewriteContext<'a>,
    indent: Indent,
    visitor_kind: &ItemVisitorKind<'b>,
    span: Span,
) -> Option<String> {
    use ItemVisitorKind::*;

    let ast::TyAlias {
        defaultness,
        ref generics,
        ref bounds,
        ref ty,
    } = *ty_alias_kind;
    let ty_opt = ty.as_ref().map(|t| &**t);
    let (ident, vis) = match visitor_kind {
        Item(i) => (i.ident, &i.vis),
        AssocTraitItem(i) | AssocImplItem(i) => (i.ident, &i.vis),
        ForeignItem(i) => (i.ident, &i.vis),
    };
    let rw_info = &TyAliasRewriteInfo(context, indent, generics, ident, span);

    // Type Aliases are formatted slightly differently depending on the context
    // in which they appear, whether they are opaque, and whether they are associated.
    // https://rustc-dev-guide.rust-lang.org/opaque-types-type-alias-impl-trait.html
    // https://github.com/rust-dev-tools/fmt-rfcs/blob/master/guide/items.md#type-aliases
    match (visitor_kind, ty_opt) {
        (Item(_), None) => {
            let op_ty = OpaqueType { bounds };
            rewrite_ty(rw_info, Some(bounds), Some(&op_ty), vis)
        }
        (Item(_), Some(ty)) => rewrite_ty(rw_info, Some(bounds), Some(&*ty), vis),
        (AssocImplItem(_), _) => {
            let result = if let Some(ast::Ty {
                kind: ast::TyKind::ImplTrait(_, ref bounds),
                ..
            }) = ty_opt
            {
                let op_ty = OpaqueType { bounds };
                rewrite_ty(rw_info, None, Some(&op_ty), &DEFAULT_VISIBILITY)
            } else {
                rewrite_ty(rw_info, None, ty.as_ref(), vis)
            }?;
            match defaultness {
                ast::Defaultness::Default(..) => Some(format!("default {}", result)),
                _ => Some(result),
            }
        }
        (AssocTraitItem(_), _) | (ForeignItem(_), _) => {
            rewrite_ty(rw_info, Some(bounds), ty.as_ref(), vis)
        }
    }
}

fn rewrite_ty<R: Rewrite>(
    rw_info: &TyAliasRewriteInfo<'_, '_>,
    generic_bounds_opt: Option<&ast::GenericBounds>,
    rhs: Option<&R>,
    vis: &ast::Visibility,
) -> Option<String> {
    let mut result = String::with_capacity(128);
    let TyAliasRewriteInfo(context, indent, generics, ident, span) = *rw_info;
    result.push_str(&format!("{}type ", format_visibility(context, vis)));
    let ident_str = rewrite_ident(context, ident);

    if generics.params.is_empty() {
        result.push_str(ident_str)
    } else {
        // 2 = `= `
        let g_shape = Shape::indented(indent, context.config)
            .offset_left(result.len())?
            .sub_width(2)?;
        let generics_str = rewrite_generics(context, ident_str, generics, g_shape)?;
        result.push_str(&generics_str);
    }

    if let Some(bounds) = generic_bounds_opt {
        if !bounds.is_empty() {
            // 2 = `: `
            let shape = Shape::indented(indent, context.config).offset_left(result.len() + 2)?;
            let type_bounds = bounds.rewrite(context, shape).map(|s| format!(": {}", s))?;
            result.push_str(&type_bounds);
        }
    }

    let where_budget = context.budget(last_line_width(&result));
    let mut option = WhereClauseOption::snuggled(&result);
    if rhs.is_none() {
        option.suppress_comma();
    }
    let where_clause_str = rewrite_where_clause(
        context,
        &generics.where_clause,
        context.config.brace_style(),
        Shape::legacy(where_budget, indent),
        false,
        "=",
        None,
        generics.span.hi(),
        option,
    )?;
    result.push_str(&where_clause_str);

    if let Some(ty) = rhs {
        // If there's a where clause, add a newline before the assignment. Otherwise just add a
        // space.
        let has_where = !generics.where_clause.predicates.is_empty();
        if has_where {
            result.push_str(&indent.to_string_with_newline(context.config));
        } else {
            result.push(' ');
        }

        let comment_span = context
            .snippet_provider
            .opt_span_before(span, "=")
            .map(|op_lo| mk_sp(generics.where_clause.span.hi(), op_lo));

        let lhs = match comment_span {
            Some(comment_span)
                if contains_comment(context.snippet_provider.span_to_snippet(comment_span)?) =>
            {
                let comment_shape = if has_where {
                    Shape::indented(indent, context.config)
                } else {
                    Shape::indented(indent, context.config)
                        .block_left(context.config.tab_spaces())?
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
            _ => format!("{}=", result),
        };

        // 1 = `;`
        let shape = Shape::indented(indent, context.config).sub_width(1)?;
        rewrite_assign_rhs(context, lhs, &*ty, shape).map(|s| s + ";")
    } else {
        Some(format!("{};", result))
    }
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
) -> Option<String> {
    let vis = format_visibility(context, &field.vis);
    let type_annotation_spacing = type_annotation_spacing(context.config);
    Some(match field.ident {
        Some(name) => format!(
            "{}{}{}:",
            vis,
            rewrite_ident(context, name),
            type_annotation_spacing.0
        ),
        None => vis.to_string(),
    })
}

impl Rewrite for ast::FieldDef {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        rewrite_struct_field(context, self, shape, 0)
    }
}

pub(crate) fn rewrite_struct_field(
    context: &RewriteContext<'_>,
    field: &ast::FieldDef,
    shape: Shape,
    lhs_max_width: usize,
) -> Option<String> {
    if contains_skip(&field.attrs) {
        return Some(context.snippet(field.span()).to_owned());
    }

    let type_annotation_spacing = type_annotation_spacing(context.config);
    let prefix = rewrite_struct_field_prefix(context, field)?;

    let attrs_str = field.attrs.rewrite(context, shape)?;
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
        .and_then(|ty_shape| field.ty.rewrite(context, ty_shape));
    if let Some(ref ty) = orig_ty {
        if !ty.contains('\n') {
            return Some(attr_prefix + &spacing + ty);
        }
    }

    let is_prefix_empty = prefix.is_empty();
    // We must use multiline. We are going to put attributes and a field on different lines.
    let field_str = rewrite_assign_rhs(context, prefix, &*field.ty, shape)?;
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
    vis: &'a ast::Visibility,
    ident: symbol::Ident,
    ty: &'a ast::Ty,
    mutability: ast::Mutability,
    expr_opt: Option<&'a ptr::P<ast::Expr>>,
    defaultness: Option<ast::Defaultness>,
    span: Span,
}

impl<'a> StaticParts<'a> {
    pub(crate) fn from_item(item: &'a ast::Item) -> Self {
        let (defaultness, prefix, ty, mutability, expr) = match item.kind {
            ast::ItemKind::Static(ref ty, mutability, ref expr) => {
                (None, "static", ty, mutability, expr)
            }
            ast::ItemKind::Const(defaultness, ref ty, ref expr) => {
                (Some(defaultness), "const", ty, ast::Mutability::Not, expr)
            }
            _ => unreachable!(),
        };
        StaticParts {
            prefix,
            vis: &item.vis,
            ident: item.ident,
            ty,
            mutability,
            expr_opt: expr.as_ref(),
            defaultness,
            span: item.span,
        }
    }

    pub(crate) fn from_trait_item(ti: &'a ast::AssocItem) -> Self {
        let (defaultness, ty, expr_opt) = match ti.kind {
            ast::AssocItemKind::Const(defaultness, ref ty, ref expr_opt) => {
                (defaultness, ty, expr_opt)
            }
            _ => unreachable!(),
        };
        StaticParts {
            prefix: "const",
            vis: &ti.vis,
            ident: ti.ident,
            ty,
            mutability: ast::Mutability::Not,
            expr_opt: expr_opt.as_ref(),
            defaultness: Some(defaultness),
            span: ti.span,
        }
    }

    pub(crate) fn from_impl_item(ii: &'a ast::AssocItem) -> Self {
        let (defaultness, ty, expr) = match ii.kind {
            ast::AssocItemKind::Const(defaultness, ref ty, ref expr) => (defaultness, ty, expr),
            _ => unreachable!(),
        };
        StaticParts {
            prefix: "const",
            vis: &ii.vis,
            ident: ii.ident,
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
    let colon = colon_spaces(context.config);
    let mut prefix = format!(
        "{}{}{} {}{}{}",
        format_visibility(context, static_parts.vis),
        static_parts.defaultness.map_or("", format_defaultness),
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

        let lhs = format!("{}{} =", prefix, ty_str);

        // 1 = ;
        let remaining_width = context.budget(offset.block_indent + 1);
        rewrite_assign_rhs_with_comments(
            context,
            &lhs,
            &**expr,
            Shape::legacy(remaining_width, offset.block_only()),
            RhsTactics::Default,
            comments_span,
            true,
        )
        .and_then(|res| recover_comment_removed(res, static_parts.span, context))
        .map(|s| if s.ends_with(';') { s } else { s + ";" })
    } else {
        Some(format!("{}{};", prefix, ty_str))
    }
}
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
        match *self {
            ast::FnRetTy::Default(_) => Some(String::new()),
            ast::FnRetTy::Ty(ref ty) => {
                if context.config.version() == Version::One
                    || context.config.indent_style() == IndentStyle::Visual
                {
                    let inner_width = shape.width.checked_sub(3)?;
                    return ty
                        .rewrite(context, Shape::legacy(inner_width, shape.indent + 3))
                        .map(|r| format!("-> {}", r));
                }

                ty.rewrite(context, shape.offset_left(3)?)
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
        .filter(|comment| !comment.is_empty())
        .map_or(String::new(), |comment| format!(" {}", comment));
    let comment_after_colon = rewrite_missing_comment(span_after_colon, shape, context)
        .filter(|comment| !comment.is_empty())
        .map_or(String::new(), |comment| format!("{} ", comment));
    (comment_before_colon, comment_after_colon)
}

impl Rewrite for ast::Param {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        let param_attrs_result = self
            .attrs
            .rewrite(context, Shape::legacy(shape.width, shape.indent))?;
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
                .rewrite(context, Shape::legacy(shape.width, shape.indent))?;
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
                let max_width = shape.width.checked_sub(overhead)?;
                if let Some(ty_str) = self
                    .ty
                    .rewrite(context, Shape::legacy(max_width, shape.indent))
                {
                    result.push_str(&ty_str);
                } else {
                    result = combine_strs_with_missing_comments(
                        context,
                        &(param_attrs_result + &shape.to_string_with_newline(context.config)),
                        param_name,
                        span,
                        shape,
                        !has_multiple_attr_lines,
                    )?;
                    result.push_str(&before_comment);
                    result.push_str(colon_spaces(context.config));
                    result.push_str(&after_comment);
                    let overhead = last_line_width(&result);
                    let max_width = shape.width.checked_sub(overhead)?;
                    let ty_str = self
                        .ty
                        .rewrite(context, Shape::legacy(max_width, shape.indent))?;
                    result.push_str(&ty_str);
                }
            }

            Some(result)
        } else {
            self.ty.rewrite(context, shape)
        }
    }
}

fn rewrite_explicit_self(
    context: &RewriteContext<'_>,
    explicit_self: &ast::ExplicitSelf,
    param_attrs: &str,
    span: Span,
    shape: Shape,
    has_multiple_attr_lines: bool,
) -> Option<String> {
    match explicit_self.node {
        ast::SelfKind::Region(lt, m) => {
            let mut_str = format_mutability(m);
            match lt {
                Some(ref l) => {
                    let lifetime_str = l.rewrite(
                        context,
                        Shape::legacy(context.config.max_width(), Indent::empty()),
                    )?;
                    Some(combine_strs_with_missing_comments(
                        context,
                        param_attrs,
                        &format!("&{} {}self", lifetime_str, mut_str),
                        span,
                        shape,
                        !has_multiple_attr_lines,
                    )?)
                }
                None => Some(combine_strs_with_missing_comments(
                    context,
                    param_attrs,
                    &format!("&{}self", mut_str),
                    span,
                    shape,
                    !has_multiple_attr_lines,
                )?),
            }
        }
        ast::SelfKind::Explicit(ref ty, mutability) => {
            let type_str = ty.rewrite(
                context,
                Shape::legacy(context.config.max_width(), Indent::empty()),
            )?;

            Some(combine_strs_with_missing_comments(
                context,
                param_attrs,
                &format!("{}self: {}", format_mutability(mutability), type_str),
                span,
                shape,
                !has_multiple_attr_lines,
            )?)
        }
        ast::SelfKind::Value(mutability) => Some(combine_strs_with_missing_comments(
            context,
            param_attrs,
            &format!("{}self", format_mutability(mutability)),
            span,
            shape,
            !has_multiple_attr_lines,
        )?),
    }
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
    if let ast::PatKind::Ident(_, ident, _) = param.pat.kind {
        ident.name != symbol::kw::Empty
    } else {
        true
    }
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
) -> Option<(String, bool, bool)> {
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
        fn_sig.generics,
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
        .rewrite(context, Shape::indented(indent, context.config))?;

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
    )?;

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

        if context.config.version() == Version::Two {
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
            if context.config.version() == Version::One
                || context.config.indent_style() == IndentStyle::Visual
            {
                let indent = if param_str.is_empty() {
                    // Aligning with non-existent params looks silly.
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
                    // Aligning with non-existent params looks silly.
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
            if context.config.version() == Version::Two {
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
            let ret_str = fd.output.rewrite(context, ret_shape)?;
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
        where_clause,
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
                mk_sp(params_span.hi(), ret_span.hi()),
                shape,
                context,
                last_line_width(&result),
            ) {
                Some(ref missing_comment) if !missing_comment.is_empty() => {
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
    Some((result, ends_with_comment, force_new_line_for_brace))
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
) -> Option<String> {
    if params.is_empty() {
        let comment = context
            .snippet(mk_sp(
                span.lo(),
                // to remove ')'
                span.hi() - BytePos(1),
            ))
            .trim();
        return Some(comment.to_owned());
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
                .rewrite(context, Shape::legacy(multi_line_budget, param_indent))
                .or_else(|| Some(context.snippet(param.span()).to_owned()))
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
            .fn_args_layout()
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
) -> Option<(usize, usize, Indent)> {
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

            return Some((one_line_budget, multi_line_budget, indent));
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
    Some((0, context.budget(used_space), new_indent))
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
) -> Option<String> {
    // FIXME: convert bounds to where-clauses where they get too big or if
    // there is a where-clause at all.

    if generics.params.is_empty() {
        return Some(ident.to_owned());
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
    where_clause: &ast::WhereClause,
    shape: Shape,
    terminator: &str,
    span_end: Option<BytePos>,
    span_end_before_where: BytePos,
    where_clause_option: WhereClauseOption,
) -> Option<String> {
    let (where_keyword, allow_single_line) = rewrite_where_keyword(
        context,
        where_clause,
        shape,
        span_end_before_where,
        where_clause_option,
    )?;

    // 1 = `,`
    let clause_shape = shape
        .block()
        .with_max_width(context.config)
        .block_left(context.config.tab_spaces())?
        .sub_width(1)?;
    let force_single_line = context.config.where_single_line()
        && where_clause.predicates.len() == 1
        && !where_clause_option.veto_single_line;

    let preds_str = rewrite_bounds_on_where_clause(
        context,
        where_clause,
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

    Some(format!("{}{}{}", where_keyword, clause_sep, preds_str))
}

/// Rewrite `where` and comment around it.
fn rewrite_where_keyword(
    context: &RewriteContext<'_>,
    where_clause: &ast::WhereClause,
    shape: Shape,
    span_end_before_where: BytePos,
    where_clause_option: WhereClauseOption,
) -> Option<(String, bool)> {
    let block_shape = shape.block().with_max_width(context.config);
    // 1 = `,`
    let clause_shape = block_shape
        .block_left(context.config.tab_spaces())?
        .sub_width(1)?;

    let comment_separator = |comment: &str, shape: Shape| {
        if comment.is_empty() {
            Cow::from("")
        } else {
            shape.indent.to_string_with_newline(context.config)
        }
    };

    let (span_before, span_after) =
        missing_span_before_after_where(span_end_before_where, where_clause);
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
        "{}{}{}where{}{}",
        starting_newline, comment_before, newline_before_where, newline_after_where, comment_after
    );
    let allow_single_line = where_clause_option.allow_single_line
        && comment_before.is_empty()
        && comment_after.is_empty();

    Some((result, allow_single_line))
}

/// Rewrite bounds on a where clause.
fn rewrite_bounds_on_where_clause(
    context: &RewriteContext<'_>,
    where_clause: &ast::WhereClause,
    shape: Shape,
    terminator: &str,
    span_end: Option<BytePos>,
    where_clause_option: WhereClauseOption,
    force_single_line: bool,
) -> Option<String> {
    let span_start = where_clause.predicates[0].span().lo();
    // If we don't have the start of the next span, then use the end of the
    // predicates, but that means we miss comments.
    let len = where_clause.predicates.len();
    let end_of_preds = where_clause.predicates[len - 1].span().hi();
    let span_end = span_end.unwrap_or(end_of_preds);
    let items = itemize_list(
        context.snippet_provider,
        where_clause.predicates.iter(),
        terminator,
        ",",
        |pred| pred.span().lo(),
        |pred| pred.span().hi(),
        |pred| pred.rewrite(context, shape),
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

    let fmt = ListFormatting::new(shape, context.config)
        .tactic(shape_tactic)
        .trailing_separator(comma_tactic)
        .preserve_newline(true);
    write_list(&items.collect::<Vec<_>>(), &fmt)
}

fn rewrite_where_clause(
    context: &RewriteContext<'_>,
    where_clause: &ast::WhereClause,
    brace_style: BraceStyle,
    shape: Shape,
    on_new_line: bool,
    terminator: &str,
    span_end: Option<BytePos>,
    span_end_before_where: BytePos,
    where_clause_option: WhereClauseOption,
) -> Option<String> {
    if where_clause.predicates.is_empty() {
        return Some(String::new());
    }

    if context.config.indent_style() == IndentStyle::Block {
        return rewrite_where_clause_rfc_style(
            context,
            where_clause,
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
    let span_start = where_clause.predicates[0].span().lo();
    // If we don't have the start of the next span, then use the end of the
    // predicates, but that means we miss comments.
    let len = where_clause.predicates.len();
    let end_of_preds = where_clause.predicates[len - 1].span().hi();
    let span_end = span_end.unwrap_or(end_of_preds);
    let items = itemize_list(
        context.snippet_provider,
        where_clause.predicates.iter(),
        terminator,
        ",",
        |pred| pred.span().lo(),
        |pred| pred.span().hi(),
        |pred| pred.rewrite(context, Shape::legacy(budget, offset)),
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
        Some(format!(
            "\n{}where {}",
            (shape.indent + extra_indent).to_string(context.config),
            preds_str
        ))
    } else {
        Some(format!(" where {}", preds_str))
    }
}

fn missing_span_before_after_where(
    before_item_span_end: BytePos,
    where_clause: &ast::WhereClause,
) -> (Span, Span) {
    let missing_span_before = mk_sp(before_item_span_end, where_clause.span.lo());
    // 5 = `where`
    let pos_after_where = where_clause.span.lo() + BytePos(5);
    let missing_span_after = mk_sp(pos_after_where, where_clause.predicates[0].span().lo());
    (missing_span_before, missing_span_after)
}

fn rewrite_comments_before_after_where(
    context: &RewriteContext<'_>,
    span_before_where: Span,
    span_after_where: Span,
    shape: Shape,
) -> Option<(String, String)> {
    let before_comment = rewrite_missing_comment(span_before_where, shape, context)?;
    let after_comment = rewrite_missing_comment(
        span_after_where,
        shape.block_indent(context.config.tab_spaces()),
        context,
    )?;
    Some((before_comment, after_comment))
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
        if let Some(result_with_comment) = combine_strs_with_missing_comments(
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
    let mut result = rewrite_generics(context, "", generics, shape)?;

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
            &generics.where_clause,
            brace_style,
            Shape::legacy(budget, offset.block_only()),
            true,
            "{",
            Some(span.hi()),
            span_end_before_where,
            option,
        )?;
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
                        context.snippet_provider.span_before(span, "{")
                    },
                ),
                shape,
                context,
            ),
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
        let attrs_str = self.attrs.rewrite(context, shape)?;
        // Drop semicolon or it will be interpreted as comment.
        // FIXME: this may be a faulty span from libsyntax.
        let span = mk_sp(self.span.lo(), self.span.hi() - BytePos(1));

        let item_str = match self.kind {
            ast::ForeignItemKind::Fn(ref fn_kind) => {
                let ast::Fn {
                    defaultness,
                    ref sig,
                    ref generics,
                    ref body,
                } = **fn_kind;
                if let Some(ref body) = body {
                    let mut visitor = FmtVisitor::from_context(context);
                    visitor.block_indent = shape.indent;
                    visitor.last_pos = self.span.lo();
                    let inner_attrs = inner_attributes(&self.attrs);
                    let fn_ctxt = visit::FnCtxt::Foreign;
                    visitor.visit_fn(
                        visit::FnKind::Fn(fn_ctxt, self.ident, &sig, &self.vis, Some(body)),
                        generics,
                        &sig.decl,
                        self.span,
                        defaultness,
                        Some(&inner_attrs),
                    );
                    Some(visitor.buffer.to_owned())
                } else {
                    rewrite_fn_base(
                        context,
                        shape.indent,
                        self.ident,
                        &FnSig::from_method_sig(&sig, generics, &self.vis),
                        span,
                        FnBraceStyle::None,
                    )
                    .map(|(s, _, _)| format!("{};", s))
                }
            }
            ast::ForeignItemKind::Static(ref ty, mutability, _) => {
                // FIXME(#21): we're dropping potential comments in between the
                // function kw here.
                let vis = format_visibility(context, &self.vis);
                let mut_str = format_mutability(mutability);
                let prefix = format!(
                    "{}static {}{}:",
                    vis,
                    mut_str,
                    rewrite_ident(context, self.ident)
                );
                // 1 = ;
                rewrite_assign_rhs(context, prefix, &**ty, shape.sub_width(1)?).map(|s| s + ";")
            }
            ast::ForeignItemKind::TyAlias(ref ty_alias) => {
                let (kind, span) = (&ItemVisitorKind::ForeignItem(&self), self.span);
                rewrite_type_alias(ty_alias, context, shape.indent, kind, span)
            }
            ast::ForeignItemKind::MacCall(ref mac) => {
                rewrite_macro(mac, None, context, shape, MacroPosition::Item)
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
}

/// Rewrite an inline mod.
/// The given shape is used to format the mod's attributes.
pub(crate) fn rewrite_mod(
    context: &RewriteContext<'_>,
    item: &ast::Item,
    attrs_shape: Shape,
) -> Option<String> {
    let mut result = String::with_capacity(32);
    result.push_str(&*format_visibility(context, &item.vis));
    result.push_str("mod ");
    result.push_str(rewrite_ident(context, item.ident));
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
        ast::ItemKind::Mod(_, ast::ModKind::Loaded(_, ast::Inline::Yes, _))
    )
}

pub(crate) fn is_use_item(item: &ast::Item) -> bool {
    matches!(item.kind, ast::ItemKind::Use(_))
}

pub(crate) fn is_extern_crate(item: &ast::Item) -> bool {
    matches!(item.kind, ast::ItemKind::ExternCrate(..))
}
