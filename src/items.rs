// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Formatting top-level items - functions, structs, enums, traits, impls.

use std::borrow::Cow;
use std::cmp::min;

use syntax::{abi, ast, ptr, symbol};
use syntax::ast::ImplItem;
use syntax::codemap::{BytePos, Span};
use syntax::visit;

use spanned::Spanned;
use codemap::{LineRangeUtils, SpanUtils};
use comment::{combine_strs_with_missing_comments, contains_comment, recover_comment_removed,
              recover_missing_comment_in_span, rewrite_missing_comment, FindUncommented};
use config::{BraceStyle, Config, Density, IndentStyle, ReturnIndent, Style};
use expr::{format_expr, is_empty_block, is_simple_block_stmt, rewrite_assign_rhs,
           rewrite_call_inner, ExprType};
use lists::{definitive_tactic, itemize_list, write_list, DefinitiveListTactic, ListFormatting,
            ListItem, ListTactic, Separator, SeparatorPlace, SeparatorTactic};
use rewrite::{Rewrite, RewriteContext};
use shape::{Indent, Shape};
use types::join_bounds;
use utils::{colon_spaces, contains_skip, end_typaram, first_line_width, format_abi,
            format_constness, format_defaultness, format_mutability, format_unsafety,
            format_visibility, is_attributes_extendable, last_line_contains_single_line_comment,
            last_line_used_width, last_line_width, mk_sp, semicolon_for_expr, starts_with_newline,
            stmt_expr, trim_newlines, trimmed_last_line_width};
use vertical::rewrite_with_alignment;
use visitor::FmtVisitor;

fn type_annotation_separator(config: &Config) -> &str {
    colon_spaces(
        config.space_before_type_annotation(),
        config.space_after_type_annotation_colon(),
    )
}

// Statements of the form
// let pat: ty = init;
impl Rewrite for ast::Local {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        debug!(
            "Local::rewrite {:?} {} {:?}",
            self,
            shape.width,
            shape.indent
        );

        skip_out_of_file_lines_range!(context, self.span);

        if contains_skip(&self.attrs) {
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
                let indent = shape.indent + last_line_width(&result) + separator.len();
                // 1 = ;
                let budget = shape.width.checked_sub(indent.width() + 1)?;
                let rewrite = ty.rewrite(context, Shape::legacy(budget, indent))?;

                infix.push_str(separator);
                infix.push_str(&rewrite);
            }

            if self.init.is_some() {
                infix.push_str(" =");
            }

            infix
        };

        result.push_str(&infix);

        if let Some(ref ex) = self.init {
            // 1 = trailing semicolon;
            let nested_shape = shape.sub_width(1)?;

            result = rewrite_assign_rhs(context, result, ex, nested_shape)?;
        }

        result.push(';');
        Some(result)
    }
}

// TODO convert to using rewrite style rather than visitor
// TODO format modules in this style
#[allow(dead_code)]
struct Item<'a> {
    keyword: &'static str,
    abi: Cow<'static, str>,
    vis: Option<&'a ast::Visibility>,
    body: Vec<BodyElement<'a>>,
    span: Span,
}

impl<'a> Item<'a> {
    fn from_foreign_mod(fm: &'a ast::ForeignMod, span: Span, config: &Config) -> Item<'a> {
        Item {
            keyword: "",
            abi: format_abi(fm.abi, config.force_explicit_abi(), true),
            vis: None,
            body: fm.items
                .iter()
                .map(|i| BodyElement::ForeignItem(i))
                .collect(),
            span: span,
        }
    }
}

enum BodyElement<'a> {
    // Stmt(&'a ast::Stmt),
    // Field(&'a ast::Field),
    // Variant(&'a ast::Variant),
    // Item(&'a ast::Item),
    ForeignItem(&'a ast::ForeignItem),
}

/// Represents a fn's signature.
pub struct FnSig<'a> {
    decl: &'a ast::FnDecl,
    generics: &'a ast::Generics,
    abi: abi::Abi,
    constness: ast::Constness,
    defaultness: ast::Defaultness,
    unsafety: ast::Unsafety,
    visibility: ast::Visibility,
}

impl<'a> FnSig<'a> {
    pub fn new(
        decl: &'a ast::FnDecl,
        generics: &'a ast::Generics,
        vis: ast::Visibility,
    ) -> FnSig<'a> {
        FnSig {
            decl: decl,
            generics: generics,
            abi: abi::Abi::Rust,
            constness: ast::Constness::NotConst,
            defaultness: ast::Defaultness::Final,
            unsafety: ast::Unsafety::Normal,
            visibility: vis,
        }
    }

    pub fn from_method_sig(
        method_sig: &'a ast::MethodSig,
        generics: &'a ast::Generics,
    ) -> FnSig<'a> {
        FnSig {
            unsafety: method_sig.unsafety,
            constness: method_sig.constness.node,
            defaultness: ast::Defaultness::Final,
            abi: method_sig.abi,
            decl: &*method_sig.decl,
            generics: generics,
            visibility: ast::Visibility::Inherited,
        }
    }

    pub fn from_fn_kind(
        fn_kind: &'a visit::FnKind,
        generics: &'a ast::Generics,
        decl: &'a ast::FnDecl,
        defualtness: ast::Defaultness,
    ) -> FnSig<'a> {
        match *fn_kind {
            visit::FnKind::ItemFn(_, unsafety, constness, abi, visibility, _) => FnSig {
                decl: decl,
                generics: generics,
                abi: abi,
                constness: constness.node,
                defaultness: defualtness,
                unsafety: unsafety,
                visibility: visibility.clone(),
            },
            visit::FnKind::Method(_, ref method_sig, vis, _) => {
                let mut fn_sig = FnSig::from_method_sig(method_sig, generics);
                fn_sig.defaultness = defualtness;
                if let Some(vis) = vis {
                    fn_sig.visibility = vis.clone();
                }
                fn_sig
            }
            _ => unreachable!(),
        }
    }

    fn to_str(&self, context: &RewriteContext) -> String {
        let mut result = String::with_capacity(128);
        // Vis defaultness constness unsafety abi.
        result.push_str(&*format_visibility(&self.visibility));
        result.push_str(format_defaultness(self.defaultness));
        result.push_str(format_constness(self.constness));
        result.push_str(format_unsafety(self.unsafety));
        result.push_str(&format_abi(
            self.abi,
            context.config.force_explicit_abi(),
            false,
        ));
        result
    }
}

impl<'a> FmtVisitor<'a> {
    fn format_item(&mut self, item: Item) {
        self.buffer.push_str(&item.abi);

        let snippet = self.snippet(item.span);
        let brace_pos = snippet.find_uncommented("{").unwrap();

        self.buffer.push_str("{");
        if !item.body.is_empty() || contains_comment(&snippet[brace_pos..]) {
            // FIXME: this skips comments between the extern keyword and the opening
            // brace.
            self.last_pos = item.span.lo() + BytePos(brace_pos as u32 + 1);
            self.block_indent = self.block_indent.block_indent(self.config);

            if item.body.is_empty() {
                self.format_missing_no_indent(item.span.hi() - BytePos(1));
                self.block_indent = self.block_indent.block_unindent(self.config);

                self.buffer
                    .push_str(&self.block_indent.to_string(self.config));
            } else {
                for item in &item.body {
                    self.format_body_element(item);
                }

                self.block_indent = self.block_indent.block_unindent(self.config);
                self.format_missing_with_indent(item.span.hi() - BytePos(1));
            }
        }

        self.buffer.push_str("}");
        self.last_pos = item.span.hi();
    }

    fn format_body_element(&mut self, element: &BodyElement) {
        match *element {
            BodyElement::ForeignItem(item) => self.format_foreign_item(item),
        }
    }

    pub fn format_foreign_mod(&mut self, fm: &ast::ForeignMod, span: Span) {
        let item = Item::from_foreign_mod(fm, span, self.config);
        self.format_item(item);
    }


    fn format_foreign_item(&mut self, item: &ast::ForeignItem) {
        let rewrite = item.rewrite(&self.get_context(), self.shape());
        self.push_rewrite(item.span(), rewrite);
        self.last_pos = item.span.hi();
    }

    pub fn rewrite_fn(
        &mut self,
        indent: Indent,
        ident: ast::Ident,
        fn_sig: &FnSig,
        span: Span,
        block: &ast::Block,
    ) -> Option<String> {
        let context = self.get_context();

        let has_body =
            !is_empty_block(block, self.codemap) || !context.config.fn_empty_single_line();
        let mut newline_brace =
            newline_for_brace(self.config, &fn_sig.generics.where_clause, has_body);

        let (mut result, force_newline_brace) =
            rewrite_fn_base(&context, indent, ident, fn_sig, span, newline_brace, true)?;

        if self.config.fn_brace_style() == BraceStyle::AlwaysNextLine || force_newline_brace {
            newline_brace = true;
        } else if last_line_width(&result) + 2 > self.shape().width {
            // 2 = ` {`
            newline_brace = true;
        } else if !result.contains('\n') {
            newline_brace = false;
        }

        // Prepare for the function body by possibly adding a newline and
        // indent.
        // FIXME we'll miss anything between the end of the signature and the
        // start of the body, but we need more spans from the compiler to solve
        // this.
        if newline_brace {
            result.push('\n');
            result.push_str(&indent.to_string(self.config));
        } else {
            result.push(' ');
        }

        self.single_line_fn(&result, block).or_else(|| Some(result))
    }

    pub fn rewrite_required_fn(
        &mut self,
        indent: Indent,
        ident: ast::Ident,
        sig: &ast::MethodSig,
        generics: &ast::Generics,
        span: Span,
    ) -> Option<String> {
        // Drop semicolon or it will be interpreted as comment.
        let span = mk_sp(span.lo(), span.hi() - BytePos(1));
        let context = self.get_context();

        let (mut result, _) = rewrite_fn_base(
            &context,
            indent,
            ident,
            &FnSig::from_method_sig(sig, generics),
            span,
            false,
            false,
        )?;

        // Re-attach semicolon
        result.push(';');

        Some(result)
    }

    fn single_line_fn(&self, fn_str: &str, block: &ast::Block) -> Option<String> {
        if fn_str.contains('\n') {
            return None;
        }

        let codemap = self.get_context().codemap;

        if self.config.fn_empty_single_line() && is_empty_block(block, codemap)
            && self.block_indent.width() + fn_str.len() + 2 <= self.config.max_width()
        {
            return Some(format!("{}{{}}", fn_str));
        }

        if self.config.fn_single_line() && is_simple_block_stmt(block, codemap) {
            let rewrite = {
                if let Some(stmt) = block.stmts.first() {
                    match stmt_expr(stmt) {
                        Some(e) => {
                            let suffix = if semicolon_for_expr(&self.get_context(), e) {
                                ";"
                            } else {
                                ""
                            };

                            format_expr(e, ExprType::Statement, &self.get_context(), self.shape())
                                .map(|s| s + suffix)
                                .or_else(|| Some(self.snippet(e.span)))
                        }
                        None => stmt.rewrite(&self.get_context(), self.shape()),
                    }
                } else {
                    None
                }
            };

            if let Some(res) = rewrite {
                let width = self.block_indent.width() + fn_str.len() + res.len() + 4;
                if !res.contains('\n') && width <= self.config.max_width() {
                    return Some(format!("{}{{ {} }}", fn_str, res));
                }
            }
        }

        None
    }

    pub fn visit_enum(
        &mut self,
        ident: ast::Ident,
        vis: &ast::Visibility,
        enum_def: &ast::EnumDef,
        generics: &ast::Generics,
        span: Span,
    ) {
        let enum_header = format_header("enum ", ident, vis);
        self.buffer.push_str(&enum_header);

        let enum_snippet = self.snippet(span);
        let brace_pos = enum_snippet.find_uncommented("{").unwrap();
        let body_start = span.lo() + BytePos(brace_pos as u32 + 1);
        let generics_str = format_generics(
            &self.get_context(),
            generics,
            "{",
            "{",
            self.config.item_brace_style(),
            enum_def.variants.is_empty(),
            self.block_indent,
            mk_sp(span.lo(), body_start),
            last_line_width(&enum_header),
        ).unwrap();
        self.buffer.push_str(&generics_str);

        self.last_pos = body_start;

        self.block_indent = self.block_indent.block_indent(self.config);
        let variant_list = self.format_variant_list(enum_def, body_start, span.hi() - BytePos(1));
        match variant_list {
            Some(ref body_str) => self.buffer.push_str(body_str),
            None => if contains_comment(&enum_snippet[brace_pos..]) {
                self.format_missing_no_indent(span.hi() - BytePos(1))
            },
        }
        self.block_indent = self.block_indent.block_unindent(self.config);

        if variant_list.is_some() || contains_comment(&enum_snippet[brace_pos..]) {
            self.buffer
                .push_str(&self.block_indent.to_string(self.config));
        }
        self.buffer.push_str("}");
        self.last_pos = span.hi();
    }

    // Format the body of an enum definition
    fn format_variant_list(
        &self,
        enum_def: &ast::EnumDef,
        body_lo: BytePos,
        body_hi: BytePos,
    ) -> Option<String> {
        if enum_def.variants.is_empty() {
            return None;
        }
        let mut result = String::with_capacity(1024);
        result.push('\n');
        let indentation = self.block_indent.to_string(self.config);
        result.push_str(&indentation);

        let items = itemize_list(
            self.codemap,
            enum_def.variants.iter(),
            "}",
            |f| if !f.node.attrs.is_empty() {
                f.node.attrs[0].span.lo()
            } else {
                f.span.lo()
            },
            |f| f.span.hi(),
            |f| self.format_variant(f),
            body_lo,
            body_hi,
            false,
        );

        let shape = self.shape().sub_width(2).unwrap();
        let fmt = ListFormatting {
            tactic: DefinitiveListTactic::Vertical,
            separator: ",",
            trailing_separator: self.config.trailing_comma(),
            separator_place: SeparatorPlace::Back,
            shape: shape,
            ends_with_newline: true,
            preserve_newline: true,
            config: self.config,
        };

        let list = write_list(&items.collect::<Vec<_>>(), &fmt)?;
        result.push_str(&list);
        result.push('\n');
        Some(result)
    }

    // Variant of an enum.
    fn format_variant(&self, field: &ast::Variant) -> Option<String> {
        if contains_skip(&field.node.attrs) {
            let lo = field.node.attrs[0].span.lo();
            let span = mk_sp(lo, field.span.hi());
            return Some(self.snippet(span));
        }

        let context = self.get_context();
        let indent = self.block_indent;
        let shape = self.shape();
        let attrs_str = field.node.attrs.rewrite(&context, shape)?;
        let lo = field
            .node
            .attrs
            .last()
            .map_or(field.span.lo(), |attr| attr.span.hi());
        let span = mk_sp(lo, field.span.lo());

        let variant_body = match field.node.data {
            ast::VariantData::Tuple(..) | ast::VariantData::Struct(..) => {
                // FIXME: Should limit the width, as we have a trailing comma
                format_struct(
                    &context,
                    "",
                    field.node.name,
                    &ast::Visibility::Inherited,
                    &field.node.data,
                    None,
                    field.span,
                    indent,
                    Some(self.config.struct_variant_width()),
                )?
            }
            ast::VariantData::Unit(..) => if let Some(ref expr) = field.node.disr_expr {
                let one_line_width =
                    field.node.name.to_string().len() + self.snippet(expr.span).len() + 3;
                if one_line_width <= shape.width {
                    format!("{} = {}", field.node.name, self.snippet(expr.span))
                } else {
                    format!(
                        "{}\n{}{}",
                        field.node.name,
                        shape
                            .indent
                            .block_indent(self.config)
                            .to_string(self.config),
                        self.snippet(expr.span)
                    )
                }
            } else {
                String::from(field.node.name.to_string())
            },
        };

        let attrs_extendable = attrs_str.is_empty()
            || (context.config.attributes_on_same_line_as_variant()
                && is_attributes_extendable(&attrs_str));
        combine_strs_with_missing_comments(
            &context,
            &attrs_str,
            &variant_body,
            span,
            shape,
            attrs_extendable,
        )
    }
}

pub fn format_impl(
    context: &RewriteContext,
    item: &ast::Item,
    offset: Indent,
    where_span_end: Option<BytePos>,
) -> Option<String> {
    if let ast::ItemKind::Impl(_, _, _, ref generics, _, ref self_ty, ref items) = item.node {
        let mut result = String::with_capacity(128);
        let ref_and_type = format_impl_ref_and_type(context, item, offset)?;
        let indent_str = offset.to_string(context.config);
        let sep = format!("\n{}", &indent_str);
        result.push_str(&ref_and_type);

        let where_budget = if result.contains('\n') {
            context.config.max_width()
        } else {
            context.budget(last_line_width(&result))
        };
        let option = WhereClauseOption::snuggled(&ref_and_type);
        let where_clause_str = rewrite_where_clause(
            context,
            &generics.where_clause,
            context.config.item_brace_style(),
            Shape::legacy(where_budget, offset.block_only()),
            context.config.where_density(),
            "{",
            where_span_end,
            self_ty.span.hi(),
            option,
        )?;

        // If there is no where clause, we may have missing comments between the trait name and
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

        if is_impl_single_line(context, items, &result, &where_clause_str, item)? {
            result.push_str(&where_clause_str);
            if where_clause_str.contains('\n') || last_line_contains_single_line_comment(&result) {
                result.push_str(&format!("{}{{{}}}", &sep, &sep));
            } else {
                result.push_str(" {}");
            }
            return Some(result);
        }

        if !where_clause_str.is_empty() && !where_clause_str.contains('\n') {
            result.push('\n');
            let width = offset.block_indent + context.config.tab_spaces() - 1;
            let where_indent = Indent::new(0, width);
            result.push_str(&where_indent.to_string(context.config));
        }
        result.push_str(&where_clause_str);

        match context.config.item_brace_style() {
            _ if last_line_contains_single_line_comment(&result) => result.push_str(&sep),
            BraceStyle::AlwaysNextLine => result.push_str(&sep),
            BraceStyle::PreferSameLine => result.push(' '),
            BraceStyle::SameLineWhere => if !where_clause_str.is_empty() {
                result.push_str(&sep);
            } else {
                result.push(' ');
            },
        }

        result.push('{');

        let snippet = context.snippet(item.span);
        let open_pos = snippet.find_uncommented("{")? + 1;

        if !items.is_empty() || contains_comment(&snippet[open_pos..]) {
            let mut visitor = FmtVisitor::from_codemap(context.parse_session, context.config);
            visitor.block_indent = offset.block_only().block_indent(context.config);
            visitor.last_pos = item.span.lo() + BytePos(open_pos as u32);

            visitor.visit_attrs(&item.attrs, ast::AttrStyle::Inner);
            for item in items {
                visitor.visit_impl_item(item);
            }

            visitor.format_missing(item.span.hi() - BytePos(1));

            let inner_indent_str = visitor.block_indent.to_string(context.config);
            let outer_indent_str = offset.block_only().to_string(context.config);

            result.push('\n');
            result.push_str(&inner_indent_str);
            result.push_str(trim_newlines(visitor.buffer.to_string().trim()));
            result.push('\n');
            result.push_str(&outer_indent_str);
        }

        if result.chars().last().unwrap() == '{' {
            result.push_str(&sep);
        }
        result.push('}');

        Some(result)
    } else {
        unreachable!();
    }
}

fn is_impl_single_line(
    context: &RewriteContext,
    items: &[ImplItem],
    result: &str,
    where_clause_str: &str,
    item: &ast::Item,
) -> Option<bool> {
    let snippet = context.snippet(item.span);
    let open_pos = snippet.find_uncommented("{")? + 1;

    Some(
        context.config.impl_empty_single_line() && items.is_empty() && !result.contains('\n')
            && result.len() + where_clause_str.len() <= context.config.max_width()
            && !contains_comment(&snippet[open_pos..]),
    )
}

fn format_impl_ref_and_type(
    context: &RewriteContext,
    item: &ast::Item,
    offset: Indent,
) -> Option<String> {
    if let ast::ItemKind::Impl(
        unsafety,
        polarity,
        defaultness,
        ref generics,
        ref trait_ref,
        ref self_ty,
        _,
    ) = item.node
    {
        let mut result = String::with_capacity(128);

        result.push_str(&format_visibility(&item.vis));
        result.push_str(format_defaultness(defaultness));
        result.push_str(format_unsafety(unsafety));
        result.push_str("impl");

        let lo = context.codemap.span_after(item.span, "impl");
        let hi = match *trait_ref {
            Some(ref tr) => tr.path.span.lo(),
            None => self_ty.span.lo(),
        };
        let shape = generics_shape_from_config(
            context.config,
            Shape::indented(offset + last_line_width(&result), context.config),
            0,
        )?;
        let one_line_budget = shape.width.checked_sub(last_line_width(&result) + 2)?;
        let generics_str =
            rewrite_generics_inner(context, generics, shape, one_line_budget, mk_sp(lo, hi))?;

        let polarity_str = if polarity == ast::ImplPolarity::Negative {
            "!"
        } else {
            ""
        };

        if let Some(ref trait_ref) = *trait_ref {
            let result_len = result.len();
            if let Some(trait_ref_str) = rewrite_trait_ref(
                context,
                trait_ref,
                offset,
                &generics_str,
                true,
                polarity_str,
                result_len,
            ) {
                result.push_str(&trait_ref_str);
            } else {
                let generics_str =
                    rewrite_generics_inner(context, generics, shape, 0, mk_sp(lo, hi))?;
                result.push_str(&rewrite_trait_ref(
                    context,
                    trait_ref,
                    offset,
                    &generics_str,
                    false,
                    polarity_str,
                    result_len,
                )?);
            }
        } else {
            result.push_str(&generics_str);
        }

        // Try to put the self type in a single line.
        // ` for`
        let trait_ref_overhead = if trait_ref.is_some() { 4 } else { 0 };
        let curly_brace_overhead = if generics.where_clause.predicates.is_empty() {
            // If there is no where clause adapt budget for type formatting to take space and curly
            // brace into account.
            match context.config.item_brace_style() {
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
                if trait_ref.is_some() {
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
        if trait_ref.is_some() {
            result.push_str("for ");
        }
        let budget = context.budget(last_line_width(&result));
        let type_offset = match context.config.where_style() {
            Style::Legacy => new_line_offset + trait_ref_overhead,
            Style::Rfc => new_line_offset,
        };
        result.push_str(&*self_ty
            .rewrite(context, Shape::legacy(budget, type_offset))?);
        Some(result)
    } else {
        unreachable!();
    }
}

fn rewrite_trait_ref(
    context: &RewriteContext,
    trait_ref: &ast::TraitRef,
    offset: Indent,
    generics_str: &str,
    retry: bool,
    polarity_str: &str,
    result_len: usize,
) -> Option<String> {
    // 1 = space between generics and trait_ref
    let used_space = 1 + polarity_str.len() + last_line_used_width(generics_str, result_len);
    let shape = Shape::indented(offset + used_space, context.config);
    if let Some(trait_ref_str) = trait_ref.rewrite(context, shape) {
        if !(retry && trait_ref_str.contains('\n')) {
            return Some(format!(
                "{} {}{}",
                generics_str,
                polarity_str,
                &trait_ref_str
            ));
        }
    }
    // We could not make enough space for trait_ref, so put it on new line.
    if !retry {
        let offset = offset.block_indent(context.config);
        let shape = Shape::indented(offset, context.config);
        let trait_ref_str = trait_ref.rewrite(context, shape)?;
        Some(format!(
            "{}\n{}{}{}",
            generics_str,
            &offset.to_string(context.config),
            polarity_str,
            &trait_ref_str
        ))
    } else {
        None
    }
}

pub fn format_struct(
    context: &RewriteContext,
    item_name: &str,
    ident: ast::Ident,
    vis: &ast::Visibility,
    struct_def: &ast::VariantData,
    generics: Option<&ast::Generics>,
    span: Span,
    offset: Indent,
    one_line_width: Option<usize>,
) -> Option<String> {
    match *struct_def {
        ast::VariantData::Unit(..) => Some(format_unit_struct(item_name, ident, vis)),
        ast::VariantData::Tuple(ref fields, _) => format_tuple_struct(
            context,
            item_name,
            ident,
            vis,
            fields,
            generics,
            span,
            offset,
        ),
        ast::VariantData::Struct(ref fields, _) => format_struct_struct(
            context,
            item_name,
            ident,
            vis,
            fields,
            generics,
            span,
            offset,
            one_line_width,
        ),
    }
}

pub fn format_trait(context: &RewriteContext, item: &ast::Item, offset: Indent) -> Option<String> {
    if let ast::ItemKind::Trait(unsafety, ref generics, ref type_param_bounds, ref trait_items) =
        item.node
    {
        let mut result = String::with_capacity(128);
        let header = format!(
            "{}{}trait {}",
            format_visibility(&item.vis),
            format_unsafety(unsafety),
            item.ident
        );

        result.push_str(&header);

        let body_lo = context.codemap.span_after(item.span, "{");

        let shape = Shape::indented(offset + last_line_width(&result), context.config);
        let generics_str =
            rewrite_generics(context, generics, shape, mk_sp(item.span.lo(), body_lo))?;
        result.push_str(&generics_str);

        let trait_bound_str = rewrite_trait_bounds(
            context,
            type_param_bounds,
            Shape::indented(offset, context.config),
        )?;
        // If the trait, generics, and trait bound cannot fit on the same line,
        // put the trait bounds on an indented new line
        if offset.width() + last_line_width(&result) + trait_bound_str.len()
            > context.config.comment_width()
        {
            result.push('\n');
            let trait_indent = offset.block_only().block_indent(context.config);
            result.push_str(&trait_indent.to_string(context.config));
        }
        result.push_str(&trait_bound_str);

        let has_body = !trait_items.is_empty();

        let where_density = if (context.config.where_density() == Density::Compressed
            && (!result.contains('\n') || context.config.fn_args_layout() == IndentStyle::Block))
            || (context.config.fn_args_layout() == IndentStyle::Block && result.is_empty())
            || (context.config.where_density() == Density::CompressedIfEmpty && !has_body
                && !result.contains('\n'))
        {
            Density::Compressed
        } else {
            Density::Tall
        };

        let where_budget = context.budget(last_line_width(&result));
        let pos_before_where = if type_param_bounds.is_empty() {
            generics.where_clause.span.lo()
        } else {
            type_param_bounds[type_param_bounds.len() - 1].span().hi()
        };
        let option = WhereClauseOption::snuggled(&generics_str);
        let where_clause_str = rewrite_where_clause(
            context,
            &generics.where_clause,
            context.config.item_brace_style(),
            Shape::legacy(where_budget, offset.block_only()),
            where_density,
            "{",
            None,
            pos_before_where,
            option,
        )?;
        // If the where clause cannot fit on the same line,
        // put the where clause on a new line
        if !where_clause_str.contains('\n')
            && last_line_width(&result) + where_clause_str.len() + offset.width()
                > context.config.comment_width()
        {
            result.push('\n');
            let width = offset.block_indent + context.config.tab_spaces() - 1;
            let where_indent = Indent::new(0, width);
            result.push_str(&where_indent.to_string(context.config));
        }
        result.push_str(&where_clause_str);

        if generics.where_clause.predicates.is_empty() {
            let item_snippet = context.snippet(item.span);
            if let Some(lo) = item_snippet.chars().position(|c| c == '/') {
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

        match context.config.item_brace_style() {
            _ if last_line_contains_single_line_comment(&result) => {
                result.push('\n');
                result.push_str(&offset.to_string(context.config));
            }
            BraceStyle::AlwaysNextLine => {
                result.push('\n');
                result.push_str(&offset.to_string(context.config));
            }
            BraceStyle::PreferSameLine => result.push(' '),
            BraceStyle::SameLineWhere => if !where_clause_str.is_empty()
                && (!trait_items.is_empty() || result.contains('\n'))
            {
                result.push('\n');
                result.push_str(&offset.to_string(context.config));
            } else {
                result.push(' ');
            },
        }
        result.push('{');

        let snippet = context.snippet(item.span);
        let open_pos = snippet.find_uncommented("{")? + 1;

        if !trait_items.is_empty() || contains_comment(&snippet[open_pos..]) {
            let mut visitor = FmtVisitor::from_codemap(context.parse_session, context.config);
            visitor.block_indent = offset.block_only().block_indent(context.config);
            visitor.last_pos = item.span.lo() + BytePos(open_pos as u32);

            for item in trait_items {
                visitor.visit_trait_item(item);
            }

            visitor.format_missing(item.span.hi() - BytePos(1));

            let inner_indent_str = visitor.block_indent.to_string(context.config);
            let outer_indent_str = offset.block_only().to_string(context.config);

            result.push('\n');
            result.push_str(&inner_indent_str);
            result.push_str(trim_newlines(visitor.buffer.to_string().trim()));
            result.push('\n');
            result.push_str(&outer_indent_str);
        } else if result.contains('\n') {
            result.push('\n');
        }

        result.push('}');
        Some(result)
    } else {
        unreachable!();
    }
}

fn format_unit_struct(item_name: &str, ident: ast::Ident, vis: &ast::Visibility) -> String {
    format!("{};", format_header(item_name, ident, vis))
}

pub fn format_struct_struct(
    context: &RewriteContext,
    item_name: &str,
    ident: ast::Ident,
    vis: &ast::Visibility,
    fields: &[ast::StructField],
    generics: Option<&ast::Generics>,
    span: Span,
    offset: Indent,
    one_line_width: Option<usize>,
) -> Option<String> {
    let mut result = String::with_capacity(1024);

    let header_str = format_header(item_name, ident, vis);
    result.push_str(&header_str);

    let body_lo = context.codemap.span_after(span, "{");

    let generics_str = match generics {
        Some(g) => format_generics(
            context,
            g,
            "{",
            "{",
            context.config.item_brace_style(),
            fields.is_empty(),
            offset,
            mk_sp(span.lo(), body_lo),
            last_line_width(&result),
        )?,
        None => {
            // 3 = ` {}`, 2 = ` {`.
            let overhead = if fields.is_empty() { 3 } else { 2 };
            if (context.config.item_brace_style() == BraceStyle::AlwaysNextLine
                && !fields.is_empty())
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
    if !generics_str.is_empty() && !generics_str.contains('\n')
        && total_width > context.config.max_width()
    {
        result.push('\n');
        result.push_str(&offset.to_string(context.config));
        result.push_str(generics_str.trim_left());
    } else {
        result.push_str(&generics_str);
    }

    if fields.is_empty() {
        let snippet = context.snippet(mk_sp(body_lo, span.hi() - BytePos(1)));
        if snippet.trim().is_empty() {
            // `struct S {}`
        } else if snippet.trim_right_matches(&[' ', '\t'][..]).ends_with('\n') {
            // fix indent
            result.push_str(snippet.trim_right());
            result.push('\n');
            result.push_str(&offset.to_string(context.config));
        } else {
            result.push_str(&snippet);
        }
        result.push('}');
        return Some(result);
    }

    // 3 = ` ` and ` }`
    let one_line_budget = context.budget(result.len() + 3 + offset.width());
    let one_line_budget =
        one_line_width.map_or(0, |one_line_width| min(one_line_width, one_line_budget));

    let items_str = rewrite_with_alignment(
        fields,
        context,
        Shape::indented(offset, context.config),
        mk_sp(body_lo, span.hi()),
        one_line_budget,
    )?;

    if !items_str.contains('\n') && !result.contains('\n') && items_str.len() <= one_line_budget {
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

fn format_tuple_struct(
    context: &RewriteContext,
    item_name: &str,
    ident: ast::Ident,
    vis: &ast::Visibility,
    fields: &[ast::StructField],
    generics: Option<&ast::Generics>,
    span: Span,
    offset: Indent,
) -> Option<String> {
    let mut result = String::with_capacity(1024);

    let header_str = format_header(item_name, ident, vis);
    result.push_str(&header_str);

    let body_lo = if fields.is_empty() {
        context.codemap.span_after(span, "(")
    } else {
        fields[0].span.lo()
    };
    let body_hi = if fields.is_empty() {
        context.codemap.span_after(span, ")")
    } else {
        // This is a dirty hack to work around a missing `)` from the span of the last field.
        let last_arg_span = fields[fields.len() - 1].span;
        if context.snippet(last_arg_span).ends_with(')') {
            last_arg_span.hi()
        } else {
            context
                .codemap
                .span_after(mk_sp(last_arg_span.hi(), span.hi()), ")")
        }
    };

    let where_clause_str = match generics {
        Some(generics) => {
            let budget = context.budget(last_line_width(&header_str));
            let shape = Shape::legacy(budget, offset);
            let g_span = mk_sp(span.lo(), body_lo);
            let generics_str = rewrite_generics(context, generics, shape, g_span)?;
            result.push_str(&generics_str);

            let where_budget = context.budget(last_line_width(&result));
            let option = WhereClauseOption::new(true, false);
            rewrite_where_clause(
                context,
                &generics.where_clause,
                context.config.item_brace_style(),
                Shape::legacy(where_budget, offset.block_only()),
                Density::Compressed,
                ";",
                None,
                body_hi,
                option,
            )?
        }
        None => "".to_owned(),
    };

    if fields.is_empty() {
        // 3 = `();`
        let used_width = last_line_used_width(&result, offset.width()) + 3;
        if used_width > context.config.max_width() {
            result.push('\n');
            result.push_str(&offset
                .block_indent(context.config)
                .to_string(context.config))
        }
        result.push('(');
        let snippet = context.snippet(mk_sp(body_lo, context.codemap.span_before(span, ")")));
        if snippet.is_empty() {
            // `struct S ()`
        } else if snippet.trim_right_matches(&[' ', '\t'][..]).ends_with('\n') {
            result.push_str(snippet.trim_right());
            result.push('\n');
            result.push_str(&offset.to_string(context.config));
        } else {
            result.push_str(&snippet);
        }
        result.push(')');
    } else {
        // 3 = `();`
        let body = rewrite_call_inner(
            context,
            "",
            &fields.iter().map(|field| field).collect::<Vec<_>>()[..],
            span,
            Shape::legacy(context.budget(last_line_width(&result) + 3), offset),
            context.config.fn_call_width(),
            false,
        )?;
        result.push_str(&body);
    }

    if !where_clause_str.is_empty() && !where_clause_str.contains('\n')
        && (result.contains('\n')
            || offset.block_indent + result.len() + where_clause_str.len() + 1
                > context.config.max_width())
    {
        // We need to put the where clause on a new line, but we didn't
        // know that earlier, so the where clause will not be indented properly.
        result.push('\n');
        result.push_str(&(offset.block_only() + (context.config.tab_spaces() - 1))
            .to_string(context.config));
    }
    result.push_str(&where_clause_str);

    Some(result)
}

pub fn rewrite_type_alias(
    context: &RewriteContext,
    indent: Indent,
    ident: ast::Ident,
    ty: &ast::Ty,
    generics: &ast::Generics,
    vis: &ast::Visibility,
    span: Span,
) -> Option<String> {
    let mut result = String::with_capacity(128);

    result.push_str(&format_visibility(vis));
    result.push_str("type ");
    result.push_str(&ident.to_string());

    // 2 = `= `
    let shape = Shape::indented(indent + result.len(), context.config).sub_width(2)?;
    let g_span = mk_sp(context.codemap.span_after(span, "type"), ty.span.lo());
    let generics_str = rewrite_generics(context, generics, shape, g_span)?;
    result.push_str(&generics_str);

    let where_budget = context.budget(last_line_width(&result));
    let option = WhereClauseOption::snuggled(&result);
    let where_clause_str = rewrite_where_clause(
        context,
        &generics.where_clause,
        context.config.item_brace_style(),
        Shape::legacy(where_budget, indent),
        context.config.where_density(),
        "=",
        Some(span.hi()),
        generics.span.hi(),
        option,
    )?;
    result.push_str(&where_clause_str);
    if where_clause_str.is_empty() {
        result.push_str(" = ");
    } else {
        result.push_str(&format!("\n{}= ", indent.to_string(context.config)));
    }

    let line_width = last_line_width(&result);
    // This checked_sub may fail as the extra space after '=' is not taken into account
    // In that case the budget is set to 0 which will make ty.rewrite retry on a new line
    let budget = context.budget(indent.width() + line_width + ";".len());
    let type_indent = indent + line_width;
    // Try to fit the type on the same line
    let ty_str = ty.rewrite(context, Shape::legacy(budget, type_indent))
        .or_else(|| {
            // The line was too short, try to put the type on the next line

            // Remove the space after '='
            result.pop();
            let type_indent = indent.block_indent(context.config);
            result.push('\n');
            result.push_str(&type_indent.to_string(context.config));
            let budget = context.budget(type_indent.width() + ";".len());
            ty.rewrite(context, Shape::legacy(budget, type_indent))
        })?;
    result.push_str(&ty_str);
    result.push_str(";");
    Some(result)
}

fn type_annotation_spacing(config: &Config) -> (&str, &str) {
    (
        if config.space_before_type_annotation() {
            " "
        } else {
            ""
        },
        if config.space_after_type_annotation_colon() {
            " "
        } else {
            ""
        },
    )
}

pub fn rewrite_struct_field_prefix(
    context: &RewriteContext,
    field: &ast::StructField,
) -> Option<String> {
    let vis = format_visibility(&field.vis);
    let type_annotation_spacing = type_annotation_spacing(context.config);
    Some(match field.ident {
        Some(name) => format!("{}{}{}:", vis, name, type_annotation_spacing.0),
        None => format!("{}", vis),
    })
}

impl Rewrite for ast::StructField {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        rewrite_struct_field(context, self, shape, 0)
    }
}

pub fn rewrite_struct_field(
    context: &RewriteContext,
    field: &ast::StructField,
    shape: Shape,
    lhs_max_width: usize,
) -> Option<String> {
    if contains_skip(&field.attrs) {
        return Some(context.snippet(mk_sp(field.attrs[0].span.lo(), field.span.hi())));
    }

    let type_annotation_spacing = type_annotation_spacing(context.config);
    let prefix = rewrite_struct_field_prefix(context, field)?;

    let attrs_str = field.attrs.rewrite(context, shape)?;
    let attrs_extendable = attrs_str.is_empty()
        || (context.config.attributes_on_same_line_as_field()
            && is_attributes_extendable(&attrs_str));
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
    let overhead = last_line_width(&attr_prefix);
    let lhs_offset = lhs_max_width.checked_sub(overhead).unwrap_or(0);
    for _ in 0..lhs_offset {
        spacing.push(' ');
    }
    // In this extreme case we will be missing a space betweeen an attribute and a field.
    if prefix.is_empty() && !attrs_str.is_empty() && attrs_extendable && spacing.is_empty() {
        spacing.push(' ');
    }
    let ty_shape = shape.offset_left(overhead + spacing.len())?;
    if let Some(ref ty) = field.ty.rewrite(context, ty_shape) {
        if !ty.contains('\n') {
            return Some(attr_prefix + &spacing + ty);
        }
    }

    // We must use multiline.
    let new_shape = shape.with_max_width(context.config);
    let ty_rewritten = field.ty.rewrite(context, new_shape)?;

    let field_str = if prefix.is_empty() {
        ty_rewritten
    } else if prefix.len() + first_line_width(&ty_rewritten) + 1 <= shape.width {
        prefix + " " + &ty_rewritten
    } else {
        let type_offset = shape.indent.block_indent(context.config);
        let nested_shape = Shape::indented(type_offset, context.config);
        let nested_ty = field.ty.rewrite(context, nested_shape)?;
        prefix + "\n" + &type_offset.to_string(context.config) + &nested_ty
    };
    combine_strs_with_missing_comments(
        context,
        &attrs_str,
        &field_str,
        missing_span,
        shape,
        attrs_extendable,
    )
}

pub fn rewrite_static(
    prefix: &str,
    vis: &ast::Visibility,
    ident: ast::Ident,
    ty: &ast::Ty,
    mutability: ast::Mutability,
    expr_opt: Option<&ptr::P<ast::Expr>>,
    offset: Indent,
    span: Span,
    context: &RewriteContext,
) -> Option<String> {
    let colon = colon_spaces(
        context.config.space_before_type_annotation(),
        context.config.space_after_type_annotation_colon(),
    );
    let prefix = format!(
        "{}{} {}{}{}",
        format_visibility(vis),
        prefix,
        format_mutability(mutability),
        ident,
        colon,
    );
    // 2 = " =".len()
    let ty_str = ty.rewrite(
        context,
        Shape::indented(offset.block_only(), context.config).offset_left(prefix.len() + 2)?,
    )?;

    if let Some(expr) = expr_opt {
        let lhs = format!("{}{} =", prefix, ty_str);
        // 1 = ;
        let remaining_width = context.budget(offset.block_indent + 1);
        rewrite_assign_rhs(
            context,
            lhs,
            expr,
            Shape::legacy(remaining_width, offset.block_only()),
        ).and_then(|res| recover_comment_removed(res, span, context))
            .map(|s| if s.ends_with(';') { s } else { s + ";" })
    } else {
        Some(format!("{}{};", prefix, ty_str))
    }
}

pub fn rewrite_associated_type(
    ident: ast::Ident,
    ty_opt: Option<&ptr::P<ast::Ty>>,
    ty_param_bounds_opt: Option<&ast::TyParamBounds>,
    context: &RewriteContext,
    indent: Indent,
) -> Option<String> {
    let prefix = format!("type {}", ident);

    let type_bounds_str = if let Some(ty_param_bounds) = ty_param_bounds_opt {
        // 2 = ": ".len()
        let shape = Shape::indented(indent, context.config).offset_left(prefix.len() + 2)?;
        let bounds: &[_] = ty_param_bounds;
        let bound_str = bounds
            .iter()
            .map(|ty_bound| ty_bound.rewrite(context, shape))
            .collect::<Option<Vec<_>>>()?;
        if !bounds.is_empty() {
            format!(": {}", join_bounds(context, shape, &bound_str))
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    if let Some(ty) = ty_opt {
        let ty_str = ty.rewrite(
            context,
            Shape::legacy(
                context.budget(indent.block_indent + prefix.len() + 2),
                indent.block_only(),
            ),
        )?;
        Some(format!("{}{} = {};", prefix, type_bounds_str, ty_str))
    } else {
        Some(format!("{}{};", prefix, type_bounds_str))
    }
}

pub fn rewrite_associated_impl_type(
    ident: ast::Ident,
    defaultness: ast::Defaultness,
    ty_opt: Option<&ptr::P<ast::Ty>>,
    ty_param_bounds_opt: Option<&ast::TyParamBounds>,
    context: &RewriteContext,
    indent: Indent,
) -> Option<String> {
    let result = rewrite_associated_type(ident, ty_opt, ty_param_bounds_opt, context, indent)?;

    match defaultness {
        ast::Defaultness::Default => Some(format!("default {}", result)),
        _ => Some(result),
    }
}

impl Rewrite for ast::FunctionRetTy {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        match *self {
            ast::FunctionRetTy::Default(_) => Some(String::new()),
            ast::FunctionRetTy::Ty(ref ty) => {
                let inner_width = shape.width.checked_sub(3)?;
                ty.rewrite(context, Shape::legacy(inner_width, shape.indent + 3))
                    .map(|r| format!("-> {}", r))
            }
        }
    }
}

fn is_empty_infer(context: &RewriteContext, ty: &ast::Ty) -> bool {
    match ty.node {
        ast::TyKind::Infer => {
            let original = context.snippet(ty.span);
            original != "_"
        }
        _ => false,
    }
}

impl Rewrite for ast::Arg {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        if is_named_arg(self) {
            let mut result = self.pat
                .rewrite(context, Shape::legacy(shape.width, shape.indent))?;

            if !is_empty_infer(context, &*self.ty) {
                if context.config.space_before_type_annotation() {
                    result.push_str(" ");
                }
                result.push_str(":");
                if context.config.space_after_type_annotation_colon() {
                    result.push_str(" ");
                }
                let overhead = last_line_width(&result);
                let max_width = shape.width.checked_sub(overhead)?;
                let ty_str = self.ty
                    .rewrite(context, Shape::legacy(max_width, shape.indent))?;
                result.push_str(&ty_str);
            }

            Some(result)
        } else {
            self.ty.rewrite(context, shape)
        }
    }
}

fn rewrite_explicit_self(
    explicit_self: &ast::ExplicitSelf,
    args: &[ast::Arg],
    context: &RewriteContext,
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
                    Some(format!("&{} {}self", lifetime_str, mut_str))
                }
                None => Some(format!("&{}self", mut_str)),
            }
        }
        ast::SelfKind::Explicit(ref ty, _) => {
            assert!(!args.is_empty(), "&[ast::Arg] shouldn't be empty.");

            let mutability = explicit_self_mutability(&args[0]);
            let type_str = ty.rewrite(
                context,
                Shape::legacy(context.config.max_width(), Indent::empty()),
            )?;

            Some(format!(
                "{}self: {}",
                format_mutability(mutability),
                type_str
            ))
        }
        ast::SelfKind::Value(_) => {
            assert!(!args.is_empty(), "&[ast::Arg] shouldn't be empty.");

            let mutability = explicit_self_mutability(&args[0]);

            Some(format!("{}self", format_mutability(mutability)))
        }
    }
}

// Hacky solution caused by absence of `Mutability` in `SelfValue` and
// `SelfExplicit` variants of `ast::ExplicitSelf_`.
fn explicit_self_mutability(arg: &ast::Arg) -> ast::Mutability {
    if let ast::PatKind::Ident(ast::BindingMode::ByValue(mutability), _, _) = arg.pat.node {
        mutability
    } else {
        unreachable!()
    }
}

pub fn span_lo_for_arg(arg: &ast::Arg) -> BytePos {
    if is_named_arg(arg) {
        arg.pat.span.lo()
    } else {
        arg.ty.span.lo()
    }
}

pub fn span_hi_for_arg(context: &RewriteContext, arg: &ast::Arg) -> BytePos {
    match arg.ty.node {
        ast::TyKind::Infer if context.snippet(arg.ty.span) == "_" => arg.ty.span.hi(),
        ast::TyKind::Infer if is_named_arg(arg) => arg.pat.span.hi(),
        _ => arg.ty.span.hi(),
    }
}

pub fn is_named_arg(arg: &ast::Arg) -> bool {
    if let ast::PatKind::Ident(_, ident, _) = arg.pat.node {
        ident.node != symbol::keywords::Invalid.ident()
    } else {
        true
    }
}

// Return type is (result, force_new_line_for_brace)
fn rewrite_fn_base(
    context: &RewriteContext,
    indent: Indent,
    ident: ast::Ident,
    fn_sig: &FnSig,
    span: Span,
    newline_brace: bool,
    has_body: bool,
) -> Option<(String, bool)> {
    let mut force_new_line_for_brace = false;

    let where_clause = &fn_sig.generics.where_clause;

    let mut result = String::with_capacity(1024);
    result.push_str(&fn_sig.to_str(context));

    // fn foo
    result.push_str("fn ");
    result.push_str(&ident.to_string());

    // Generics.
    let overhead = if has_body && !newline_brace {
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
        indent: indent,
        offset: used_width,
    };
    let fd = fn_sig.decl;
    let g_span = mk_sp(span.lo(), fd.output.span().lo());
    let generics_str = rewrite_generics(context, fn_sig.generics, shape, g_span)?;
    result.push_str(&generics_str);

    let snuggle_angle_bracket = generics_str
        .lines()
        .last()
        .map_or(false, |l| l.trim_left().len() == 1);

    // Note that the width and indent don't really matter, we'll re-layout the
    // return type later anyway.
    let ret_str = fd.output
        .rewrite(context, Shape::indented(indent, context.config))?;

    let multi_line_ret_str = ret_str.contains('\n');
    let ret_str_len = if multi_line_ret_str { 0 } else { ret_str.len() };

    // Args.
    let (one_line_budget, multi_line_budget, mut arg_indent) = compute_budgets_for_args(
        context,
        &result,
        indent,
        ret_str_len,
        newline_brace,
        has_body,
        multi_line_ret_str,
    )?;

    debug!(
        "rewrite_fn_base: one_line_budget: {}, multi_line_budget: {}, arg_indent: {:?}",
        one_line_budget,
        multi_line_budget,
        arg_indent
    );

    // Check if vertical layout was forced.
    if one_line_budget == 0 {
        if snuggle_angle_bracket {
            result.push('(');
        } else if context.config.fn_args_paren_newline() {
            result.push('\n');
            result.push_str(&arg_indent.to_string(context.config));
            if context.config.fn_args_layout() == IndentStyle::Visual {
                arg_indent = arg_indent + 1; // extra space for `(`
            }
            result.push('(');
        } else {
            result.push_str("(");
            if context.config.fn_args_layout() == IndentStyle::Visual {
                result.push('\n');
                result.push_str(&arg_indent.to_string(context.config));
            }
        }
    } else {
        result.push('(');
    }
    if context.config.spaces_within_parens() && !fd.inputs.is_empty() && result.ends_with('(') {
        result.push(' ')
    }

    // A conservative estimation, to goal is to be over all parens in generics
    let args_start = fn_sig
        .generics
        .ty_params
        .last()
        .map_or(span.lo(), |tp| end_typaram(tp));
    let args_end = if fd.inputs.is_empty() {
        context
            .codemap
            .span_after(mk_sp(args_start, span.hi()), ")")
    } else {
        let last_span = mk_sp(fd.inputs[fd.inputs.len() - 1].span().hi(), span.hi());
        context.codemap.span_after(last_span, ")")
    };
    let args_span = mk_sp(
        context
            .codemap
            .span_after(mk_sp(args_start, span.hi()), "("),
        args_end,
    );
    let arg_str = rewrite_args(
        context,
        &fd.inputs,
        fd.get_self().as_ref(),
        one_line_budget,
        multi_line_budget,
        indent,
        arg_indent,
        args_span,
        fd.variadic,
        generics_str.contains('\n'),
    )?;

    let put_args_in_block = match context.config.fn_args_layout() {
        IndentStyle::Block => arg_str.contains('\n') || arg_str.len() > one_line_budget,
        _ => false,
    } && !fd.inputs.is_empty();

    let mut args_last_line_contains_comment = false;
    if put_args_in_block {
        arg_indent = indent.block_indent(context.config);
        result.push('\n');
        result.push_str(&arg_indent.to_string(context.config));
        result.push_str(&arg_str);
        result.push('\n');
        result.push_str(&indent.to_string(context.config));
        result.push(')');
    } else {
        result.push_str(&arg_str);
        let used_width = last_line_used_width(&result, indent.width()) + first_line_width(&ret_str);
        // Put the closing brace on the next line if it overflows the max width.
        // 1 = `)`
        if fd.inputs.is_empty() && used_width + 1 > context.config.max_width() {
            result.push('\n');
        }
        if context.config.spaces_within_parens() && !fd.inputs.is_empty() {
            result.push(' ')
        }
        // If the last line of args contains comment, we cannot put the closing paren
        // on the same line.
        if arg_str
            .lines()
            .last()
            .map_or(false, |last_line| last_line.contains("//"))
        {
            args_last_line_contains_comment = true;
            result.push('\n');
            result.push_str(&arg_indent.to_string(context.config));
        }
        result.push(')');
    }

    // Return type.
    if let ast::FunctionRetTy::Ty(..) = fd.output {
        let ret_should_indent = match context.config.fn_args_layout() {
            // If our args are block layout then we surely must have space.
            IndentStyle::Block if put_args_in_block || fd.inputs.is_empty() => false,
            _ if args_last_line_contains_comment => false,
            _ if result.contains('\n') || multi_line_ret_str => true,
            _ => {
                // If the return type would push over the max width, then put the return type on
                // a new line. With the +1 for the signature length an additional space between
                // the closing parenthesis of the argument and the arrow '->' is considered.
                let mut sig_length = result.len() + indent.width() + ret_str_len + 1;

                // If there is no where clause, take into account the space after the return type
                // and the brace.
                if where_clause.predicates.is_empty() {
                    sig_length += 2;
                }

                sig_length > context.config.max_width()
            }
        };
        let ret_indent = if ret_should_indent {
            let indent = match context.config.fn_return_indent() {
                ReturnIndent::WithWhereClause => indent + 4,
                // Aligning with non-existent args looks silly.
                _ if arg_str.is_empty() => {
                    force_new_line_for_brace = true;
                    indent + 4
                }
                // FIXME: we might want to check that using the arg indent
                // doesn't blow our budget, and if it does, then fallback to
                // the where clause indent.
                _ => arg_indent,
            };

            result.push('\n');
            result.push_str(&indent.to_string(context.config));
            indent
        } else {
            result.push(' ');
            Indent::new(indent.block_indent, last_line_width(&result))
        };

        if multi_line_ret_str || ret_should_indent {
            // Now that we know the proper indent and width, we need to
            // re-layout the return type.
            let ret_str = fd.output
                .rewrite(context, Shape::indented(ret_indent, context.config))?;
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

    let should_compress_where = match context.config.where_density() {
        Density::Compressed => !result.contains('\n'),
        Density::CompressedIfEmpty => !has_body && !result.contains('\n'),
        _ => false,
    };

    let pos_before_where = match fd.output {
        ast::FunctionRetTy::Default(..) => args_span.hi(),
        ast::FunctionRetTy::Ty(ref ty) => ty.span.hi(),
    };

    if where_clause.predicates.len() == 1 && should_compress_where {
        let budget = context.budget(last_line_used_width(&result, indent.width()));
        if let Some(where_clause_str) = rewrite_where_clause(
            context,
            where_clause,
            context.config.fn_brace_style(),
            Shape::legacy(budget, indent),
            Density::Compressed,
            "{",
            Some(span.hi()),
            pos_before_where,
            WhereClauseOption::compressed(),
        ) {
            result.push_str(&where_clause_str);
            force_new_line_for_brace |= last_line_contains_single_line_comment(&result);
            return Some((result, force_new_line_for_brace));
        }
    }

    let option = WhereClauseOption::new(!has_body, put_args_in_block && ret_str.is_empty());
    let where_clause_str = rewrite_where_clause(
        context,
        where_clause,
        context.config.fn_brace_style(),
        Shape::indented(indent, context.config),
        Density::Tall,
        "{",
        Some(span.hi()),
        pos_before_where,
        option,
    )?;
    // If there are neither where clause nor return type, we may be missing comments between
    // args and `{`.
    if where_clause_str.is_empty() {
        if let ast::FunctionRetTy::Default(ret_span) = fd.output {
            match recover_missing_comment_in_span(
                mk_sp(args_span.hi(), ret_span.hi()),
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

    force_new_line_for_brace |= last_line_contains_single_line_comment(&result);
    Some((result, force_new_line_for_brace))
}

#[derive(Copy, Clone)]
struct WhereClauseOption {
    suppress_comma: bool, // Force no trailing comma
    snuggle: bool,        // Do not insert newline before `where`
    compress_where: bool, // Try single line where clause instead of vertical layout
}

impl WhereClauseOption {
    pub fn new(suppress_comma: bool, snuggle: bool) -> WhereClauseOption {
        WhereClauseOption {
            suppress_comma: suppress_comma,
            snuggle: snuggle,
            compress_where: false,
        }
    }

    pub fn compressed() -> WhereClauseOption {
        WhereClauseOption {
            suppress_comma: true,
            snuggle: false,
            compress_where: true,
        }
    }

    pub fn snuggled(current: &str) -> WhereClauseOption {
        WhereClauseOption {
            suppress_comma: false,
            snuggle: trimmed_last_line_width(current) == 1,
            compress_where: false,
        }
    }
}

fn rewrite_args(
    context: &RewriteContext,
    args: &[ast::Arg],
    explicit_self: Option<&ast::ExplicitSelf>,
    one_line_budget: usize,
    multi_line_budget: usize,
    indent: Indent,
    arg_indent: Indent,
    span: Span,
    variadic: bool,
    generics_str_contains_newline: bool,
) -> Option<String> {
    let mut arg_item_strs = args.iter()
        .map(|arg| {
            arg.rewrite(context, Shape::legacy(multi_line_budget, arg_indent))
        })
        .collect::<Option<Vec<_>>>()?;

    // Account for sugary self.
    // FIXME: the comment for the self argument is dropped. This is blocked
    // on rust issue #27522.
    let min_args = explicit_self
        .and_then(|explicit_self| {
            rewrite_explicit_self(explicit_self, args, context)
        })
        .map_or(1, |self_str| {
            arg_item_strs[0] = self_str;
            2
        });

    // Comments between args.
    let mut arg_items = Vec::new();
    if min_args == 2 {
        arg_items.push(ListItem::from_str(""));
    }

    // FIXME(#21): if there are no args, there might still be a comment, but
    // without spans for the comment or parens, there is no chance of
    // getting it right. You also don't get to put a comment on self, unless
    // it is explicit.
    if args.len() >= min_args || variadic {
        let comment_span_start = if min_args == 2 {
            let second_arg_start = if arg_has_pattern(&args[1]) {
                args[1].pat.span.lo()
            } else {
                args[1].ty.span.lo()
            };
            let reduced_span = mk_sp(span.lo(), second_arg_start);

            context.codemap.span_after_last(reduced_span, ",")
        } else {
            span.lo()
        };

        enum ArgumentKind<'a> {
            Regular(&'a ast::Arg),
            Variadic(BytePos),
        }

        let variadic_arg = if variadic {
            let variadic_span = mk_sp(args.last().unwrap().ty.span.hi(), span.hi());
            let variadic_start = context.codemap.span_after(variadic_span, "...") - BytePos(3);
            Some(ArgumentKind::Variadic(variadic_start))
        } else {
            None
        };

        let more_items = itemize_list(
            context.codemap,
            args[min_args - 1..]
                .iter()
                .map(ArgumentKind::Regular)
                .chain(variadic_arg),
            ")",
            |arg| match *arg {
                ArgumentKind::Regular(arg) => span_lo_for_arg(arg),
                ArgumentKind::Variadic(start) => start,
            },
            |arg| match *arg {
                ArgumentKind::Regular(arg) => arg.ty.span.hi(),
                ArgumentKind::Variadic(start) => start + BytePos(3),
            },
            |arg| match *arg {
                ArgumentKind::Regular(..) => None,
                ArgumentKind::Variadic(..) => Some("...".to_owned()),
            },
            comment_span_start,
            span.hi(),
            false,
        );

        arg_items.extend(more_items);
    }

    let fits_in_one_line = !generics_str_contains_newline
        && (arg_items.is_empty()
            || arg_items.len() == 1 && arg_item_strs[0].len() <= one_line_budget);

    for (item, arg) in arg_items.iter_mut().zip(arg_item_strs) {
        item.item = Some(arg);
    }

    let last_line_ends_with_comment = arg_items
        .iter()
        .last()
        .and_then(|item| item.post_comment.as_ref())
        .map_or(false, |s| s.trim().starts_with("//"));

    let (indent, trailing_comma) = match context.config.fn_args_layout() {
        IndentStyle::Block if fits_in_one_line => {
            (indent.block_indent(context.config), SeparatorTactic::Never)
        }
        IndentStyle::Block => (
            indent.block_indent(context.config),
            context.config.trailing_comma(),
        ),
        IndentStyle::Visual if last_line_ends_with_comment => {
            (arg_indent, context.config.trailing_comma())
        }
        IndentStyle::Visual => (arg_indent, SeparatorTactic::Never),
    };

    let tactic = definitive_tactic(
        &arg_items,
        context.config.fn_args_density().to_list_tactic(),
        Separator::Comma,
        one_line_budget,
    );
    let budget = match tactic {
        DefinitiveListTactic::Horizontal => one_line_budget,
        _ => multi_line_budget,
    };

    debug!("rewrite_args: budget: {}, tactic: {:?}", budget, tactic);

    let fmt = ListFormatting {
        tactic: tactic,
        separator: ",",
        trailing_separator: if variadic {
            SeparatorTactic::Never
        } else {
            trailing_comma
        },
        separator_place: SeparatorPlace::Back,
        shape: Shape::legacy(budget, indent),
        ends_with_newline: tactic.ends_with_newline(context.config.fn_args_layout()),
        preserve_newline: true,
        config: context.config,
    };

    write_list(&arg_items, &fmt)
}

fn arg_has_pattern(arg: &ast::Arg) -> bool {
    if let ast::PatKind::Ident(_, ident, _) = arg.pat.node {
        ident.node != symbol::keywords::Invalid.ident()
    } else {
        true
    }
}

fn compute_budgets_for_args(
    context: &RewriteContext,
    result: &str,
    indent: Indent,
    ret_str_len: usize,
    newline_brace: bool,
    has_braces: bool,
    force_vertical_layout: bool,
) -> Option<((usize, usize, Indent))> {
    debug!(
        "compute_budgets_for_args {} {:?}, {}, {}",
        result.len(),
        indent,
        ret_str_len,
        newline_brace
    );
    // Try keeping everything on the same line.
    if !result.contains('\n') && !force_vertical_layout {
        // 2 = `()`, 3 = `() `, space is before ret_string.
        let overhead = if ret_str_len == 0 { 2 } else { 3 };
        let mut used_space = indent.width() + result.len() + ret_str_len + overhead;
        if has_braces {
            if !newline_brace {
                // 2 = `{}`
                used_space += 2;
            }
        } else {
            // 1 = `;`
            used_space += 1;
        }
        let one_line_budget = context.budget(used_space);

        if one_line_budget > 0 {
            // 4 = "() {".len()
            let (indent, multi_line_budget) = match context.config.fn_args_layout() {
                IndentStyle::Block => {
                    let indent = indent.block_indent(context.config);
                    (indent, context.budget(indent.width() + 1))
                }
                IndentStyle::Visual => {
                    let indent = indent + result.len() + 1;
                    let multi_line_overhead = indent.width() + if newline_brace { 2 } else { 4 };
                    (indent, context.budget(multi_line_overhead))
                }
            };

            return Some((one_line_budget, multi_line_budget, indent));
        }
    }

    // Didn't work. we must force vertical layout and put args on a newline.
    let new_indent = indent.block_indent(context.config);
    let used_space = match context.config.fn_args_layout() {
        // 1 = `,`
        IndentStyle::Block => new_indent.width() + 1,
        // Account for `)` and possibly ` {`.
        IndentStyle::Visual => new_indent.width() + if ret_str_len == 0 { 1 } else { 3 },
    };
    Some((0, context.budget(used_space), new_indent))
}

fn newline_for_brace(config: &Config, where_clause: &ast::WhereClause, has_body: bool) -> bool {
    match (config.fn_brace_style(), config.where_density()) {
        (BraceStyle::AlwaysNextLine, _) => true,
        (_, Density::Compressed) if where_clause.predicates.len() == 1 => false,
        (_, Density::CompressedIfEmpty) if where_clause.predicates.len() == 1 && !has_body => false,
        (BraceStyle::SameLineWhere, _) if !where_clause.predicates.is_empty() => true,
        _ => false,
    }
}

fn rewrite_generics(
    context: &RewriteContext,
    generics: &ast::Generics,
    shape: Shape,
    span: Span,
) -> Option<String> {
    let g_shape = generics_shape_from_config(context.config, shape, 0)?;
    let one_line_width = shape.width.checked_sub(2).unwrap_or(0);
    rewrite_generics_inner(context, generics, g_shape, one_line_width, span).or_else(|| {
        rewrite_generics_inner(context, generics, g_shape, 0, span)
    })
}

fn rewrite_generics_inner(
    context: &RewriteContext,
    generics: &ast::Generics,
    shape: Shape,
    one_line_width: usize,
    span: Span,
) -> Option<String> {
    // FIXME: convert bounds to where clauses where they get too big or if
    // there is a where clause at all.

    // Wrapper type
    enum GenericsArg<'a> {
        Lifetime(&'a ast::LifetimeDef),
        TyParam(&'a ast::TyParam),
    }
    impl<'a> Rewrite for GenericsArg<'a> {
        fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
            match *self {
                GenericsArg::Lifetime(ref lifetime) => lifetime.rewrite(context, shape),
                GenericsArg::TyParam(ref ty) => ty.rewrite(context, shape),
            }
        }
    }
    impl<'a> Spanned for GenericsArg<'a> {
        fn span(&self) -> Span {
            match *self {
                GenericsArg::Lifetime(ref lifetime) => lifetime.span(),
                GenericsArg::TyParam(ref ty) => ty.span(),
            }
        }
    }

    if generics.lifetimes.is_empty() && generics.ty_params.is_empty() {
        return Some(String::new());
    }

    let generics_args = generics
        .lifetimes
        .iter()
        .map(|lt| GenericsArg::Lifetime(lt))
        .chain(generics.ty_params.iter().map(|ty| GenericsArg::TyParam(ty)));
    let items = itemize_list(
        context.codemap,
        generics_args,
        ">",
        |arg| arg.span().lo(),
        |arg| arg.span().hi(),
        |arg| arg.rewrite(context, shape),
        context.codemap.span_after(span, "<"),
        span.hi(),
        false,
    );
    format_generics_item_list(context, items, shape, one_line_width)
}

pub fn generics_shape_from_config(config: &Config, shape: Shape, offset: usize) -> Option<Shape> {
    match config.generics_indent() {
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

pub fn format_generics_item_list<I>(
    context: &RewriteContext,
    items: I,
    shape: Shape,
    one_line_budget: usize,
) -> Option<String>
where
    I: Iterator<Item = ListItem>,
{
    let item_vec = items.collect::<Vec<_>>();

    let tactic = definitive_tactic(
        &item_vec,
        ListTactic::HorizontalVertical,
        Separator::Comma,
        one_line_budget,
    );
    let fmt = ListFormatting {
        tactic: tactic,
        separator: ",",
        trailing_separator: if context.config.generics_indent() == IndentStyle::Visual {
            SeparatorTactic::Never
        } else {
            context.config.trailing_comma()
        },
        separator_place: SeparatorPlace::Back,
        shape: shape,
        ends_with_newline: tactic.ends_with_newline(context.config.generics_indent()),
        preserve_newline: true,
        config: context.config,
    };

    let list_str = write_list(&item_vec, &fmt)?;

    Some(wrap_generics_with_angle_brackets(
        context,
        &list_str,
        shape.indent,
    ))
}

pub fn wrap_generics_with_angle_brackets(
    context: &RewriteContext,
    list_str: &str,
    list_offset: Indent,
) -> String {
    if context.config.generics_indent() == IndentStyle::Block
        && (list_str.contains('\n') || list_str.ends_with(','))
    {
        format!(
            "<\n{}{}\n{}>",
            list_offset.to_string(context.config),
            list_str,
            list_offset
                .block_unindent(context.config)
                .to_string(context.config)
        )
    } else if context.config.spaces_within_angle_brackets() {
        format!("< {} >", list_str)
    } else {
        format!("<{}>", list_str)
    }
}

fn rewrite_trait_bounds(
    context: &RewriteContext,
    type_param_bounds: &ast::TyParamBounds,
    shape: Shape,
) -> Option<String> {
    let bounds: &[_] = type_param_bounds;

    if bounds.is_empty() {
        return Some(String::new());
    }
    let bound_str = bounds
        .iter()
        .map(|ty_bound| ty_bound.rewrite(context, shape))
        .collect::<Option<Vec<_>>>()?;
    Some(format!(": {}", join_bounds(context, shape, &bound_str)))
}

fn rewrite_where_clause_rfc_style(
    context: &RewriteContext,
    where_clause: &ast::WhereClause,
    shape: Shape,
    terminator: &str,
    span_end: Option<BytePos>,
    span_end_before_where: BytePos,
    where_clause_option: WhereClauseOption,
) -> Option<String> {
    let block_shape = shape.block().with_max_width(context.config);

    let (span_before, span_after) =
        missing_span_before_after_where(span_end_before_where, where_clause);
    let (comment_before, comment_after) =
        rewrite_comments_before_after_where(context, span_before, span_after, shape)?;

    let starting_newline = if where_clause_option.snuggle && comment_before.is_empty() {
        " ".to_owned()
    } else {
        "\n".to_owned() + &block_shape.indent.to_string(context.config)
    };

    let clause_shape = block_shape.block_left(context.config.tab_spaces())?;
    // 1 = `,`
    let clause_shape = clause_shape.sub_width(1)?;
    // each clause on one line, trailing comma (except if suppress_comma)
    let span_start = where_clause.predicates[0].span().lo();
    // If we don't have the start of the next span, then use the end of the
    // predicates, but that means we miss comments.
    let len = where_clause.predicates.len();
    let end_of_preds = where_clause.predicates[len - 1].span().hi();
    let span_end = span_end.unwrap_or(end_of_preds);
    let items = itemize_list(
        context.codemap,
        where_clause.predicates.iter(),
        terminator,
        |pred| pred.span().lo(),
        |pred| pred.span().hi(),
        |pred| pred.rewrite(context, clause_shape),
        span_start,
        span_end,
        false,
    );
    let comma_tactic = if where_clause_option.suppress_comma {
        SeparatorTactic::Never
    } else {
        context.config.trailing_comma()
    };

    let fmt = ListFormatting {
        tactic: DefinitiveListTactic::Vertical,
        separator: ",",
        trailing_separator: comma_tactic,
        separator_place: SeparatorPlace::Back,
        shape: clause_shape,
        ends_with_newline: true,
        preserve_newline: true,
        config: context.config,
    };
    let preds_str = write_list(&items.collect::<Vec<_>>(), &fmt)?;

    let comment_separator = |comment: &str, shape: Shape| if comment.is_empty() {
        String::new()
    } else {
        format!("\n{}", shape.indent.to_string(context.config))
    };
    let newline_before_where = comment_separator(&comment_before, shape);
    let newline_after_where = comment_separator(&comment_after, clause_shape);

    // 6 = `where `
    let clause_sep = if where_clause_option.compress_where && comment_before.is_empty()
        && comment_after.is_empty() && !preds_str.contains('\n')
        && 6 + preds_str.len() <= shape.width
    {
        String::from(" ")
    } else {
        format!("\n{}", clause_shape.indent.to_string(context.config))
    };
    Some(format!(
        "{}{}{}where{}{}{}{}",
        starting_newline,
        comment_before,
        newline_before_where,
        newline_after_where,
        comment_after,
        clause_sep,
        preds_str
    ))
}

fn rewrite_where_clause(
    context: &RewriteContext,
    where_clause: &ast::WhereClause,
    brace_style: BraceStyle,
    shape: Shape,
    density: Density,
    terminator: &str,
    span_end: Option<BytePos>,
    span_end_before_where: BytePos,
    where_clause_option: WhereClauseOption,
) -> Option<String> {
    if where_clause.predicates.is_empty() {
        return Some(String::new());
    }

    if context.config.where_style() == Style::Rfc {
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

    let offset = match context.config.where_pred_indent() {
        IndentStyle::Block => shape.indent + extra_indent.block_indent(context.config),
        // 6 = "where ".len()
        IndentStyle::Visual => shape.indent + extra_indent + 6,
    };
    // FIXME: if where_pred_indent != Visual, then the budgets below might
    // be out by a char or two.

    let budget = context.config.max_width() - offset.width();
    let span_start = where_clause.predicates[0].span().lo();
    // If we don't have the start of the next span, then use the end of the
    // predicates, but that means we miss comments.
    let len = where_clause.predicates.len();
    let end_of_preds = where_clause.predicates[len - 1].span().hi();
    let span_end = span_end.unwrap_or(end_of_preds);
    let items = itemize_list(
        context.codemap,
        where_clause.predicates.iter(),
        terminator,
        |pred| pred.span().lo(),
        |pred| pred.span().hi(),
        |pred| pred.rewrite(context, Shape::legacy(budget, offset)),
        span_start,
        span_end,
        false,
    );
    let item_vec = items.collect::<Vec<_>>();
    // FIXME: we don't need to collect here if the where_layout isn't
    // HorizontalVertical.
    let tactic = definitive_tactic(
        &item_vec,
        context.config.where_layout(),
        Separator::Comma,
        budget,
    );

    let mut comma_tactic = context.config.trailing_comma();
    // Kind of a hack because we don't usually have trailing commas in where clauses.
    if comma_tactic == SeparatorTactic::Vertical || where_clause_option.suppress_comma {
        comma_tactic = SeparatorTactic::Never;
    }

    let fmt = ListFormatting {
        tactic: tactic,
        separator: ",",
        trailing_separator: comma_tactic,
        separator_place: SeparatorPlace::Back,
        shape: Shape::legacy(budget, offset),
        ends_with_newline: tactic.ends_with_newline(context.config.where_pred_indent()),
        preserve_newline: true,
        config: context.config,
    };
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
    if density == Density::Tall || preds_str.contains('\n')
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
    context: &RewriteContext,
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

fn format_header(item_name: &str, ident: ast::Ident, vis: &ast::Visibility) -> String {
    format!("{}{}{}", format_visibility(vis), item_name, ident)
}

fn format_generics(
    context: &RewriteContext,
    generics: &ast::Generics,
    opener: &str,
    terminator: &str,
    brace_style: BraceStyle,
    force_same_line_brace: bool,
    offset: Indent,
    span: Span,
    used_width: usize,
) -> Option<String> {
    let shape = Shape::legacy(context.budget(used_width + offset.width()), offset);
    let mut result = rewrite_generics(context, generics, shape, span)?;

    let same_line_brace = if !generics.where_clause.predicates.is_empty() || result.contains('\n') {
        let budget = context.budget(last_line_used_width(&result, offset.width()));
        let option = WhereClauseOption::snuggled(&result);
        let where_clause_str = rewrite_where_clause(
            context,
            &generics.where_clause,
            brace_style,
            Shape::legacy(budget, offset.block_only()),
            Density::Tall,
            terminator,
            Some(span.hi()),
            generics.span.hi(),
            option,
        )?;
        result.push_str(&where_clause_str);
        force_same_line_brace || brace_style == BraceStyle::PreferSameLine
            || (generics.where_clause.predicates.is_empty()
                && trimmed_last_line_width(&result) == 1)
    } else {
        force_same_line_brace || trimmed_last_line_width(&result) == 1
            || brace_style != BraceStyle::AlwaysNextLine
    };
    let total_used_width = last_line_used_width(&result, used_width);
    let remaining_budget = context.budget(total_used_width);
    // If the same line brace if forced, it indicates that we are rewriting an item with empty body,
    // and hence we take the closer into account as well for one line budget.
    // We assume that the closer has the same length as the opener.
    let overhead = if force_same_line_brace {
        1 + opener.len() + opener.len()
    } else {
        1 + opener.len()
    };
    let forbid_same_line_brace = overhead > remaining_budget;
    if !forbid_same_line_brace && same_line_brace {
        result.push(' ');
    } else {
        result.push('\n');
        result.push_str(&offset.block_only().to_string(context.config));
    }
    result.push_str(opener);

    Some(result)
}

impl Rewrite for ast::ForeignItem {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        let attrs_str = self.attrs.rewrite(context, shape)?;
        // Drop semicolon or it will be interpreted as comment.
        // FIXME: this may be a faulty span from libsyntax.
        let span = mk_sp(self.span.lo(), self.span.hi() - BytePos(1));

        let item_str = match self.node {
            ast::ForeignItemKind::Fn(ref fn_decl, ref generics) => {
                rewrite_fn_base(
                    context,
                    shape.indent,
                    self.ident,
                    &FnSig::new(fn_decl, generics, self.vis.clone()),
                    span,
                    false,
                    false,
                ).map(|(s, _)| format!("{};", s))
            }
            ast::ForeignItemKind::Static(ref ty, is_mutable) => {
                // FIXME(#21): we're dropping potential comments in between the
                // function keywords here.
                let vis = format_visibility(&self.vis);
                let mut_str = if is_mutable { "mut " } else { "" };
                let prefix = format!("{}static {}{}:", vis, mut_str, self.ident);
                // 1 = ;
                let shape = shape.sub_width(1)?;
                ty.rewrite(context, shape).map(|ty_str| {
                    // 1 = space between prefix and type.
                    let sep = if prefix.len() + ty_str.len() + 1 <= shape.width {
                        String::from(" ")
                    } else {
                        let nested_indent = shape.indent.block_indent(context.config);
                        format!("\n{}", nested_indent.to_string(context.config))
                    };
                    format!("{}{}{};", prefix, sep, ty_str)
                })
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
