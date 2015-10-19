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

use Indent;
use utils::{format_mutability, format_visibility, contains_skip, span_after, end_typaram, wrap_str};
use lists::{write_list, itemize_list, ListItem, ListFormatting, SeparatorTactic, ListTactic,
            DefinitiveListTactic, definitive_tactic};
use expr::rewrite_assign_rhs;
use comment::FindUncommented;
use visitor::FmtVisitor;
use rewrite::{Rewrite, RewriteContext};
use config::{Config, BlockIndentStyle, Density, ReturnIndent, BraceStyle, StructLitStyle};

use syntax::{ast, abi};
use syntax::codemap::{self, Span, BytePos};
use syntax::print::pprust;
use syntax::parse::token;

impl<'a> FmtVisitor<'a> {
    pub fn visit_let(&mut self, local: &ast::Local, span: Span) {
        self.format_missing_with_indent(span.lo);

        // String that is placed within the assignment pattern and expression.
        let infix = {
            let mut infix = String::new();

            if let Some(ref ty) = local.ty {
                // 2 = ": ".len()
                let offset = self.block_indent + 2;
                let width = self.config.max_width - offset.width();
                let rewrite = ty.rewrite(&self.get_context(), width, offset);

                match rewrite {
                    Some(result) => {
                        infix.push_str(": ");
                        infix.push_str(&result);
                    }
                    None => return,
                }
            }

            if local.init.is_some() {
                infix.push_str(" =");
            }

            infix
        };

        // New scope so we drop the borrow of self (context) in time to mutably
        // borrow self to mutate its buffer.
        let result = {
            let context = self.get_context();
            let mut result = "let ".to_owned();
            let pattern_offset = self.block_indent + result.len() + infix.len();
            // 1 = ;
            let pattern_width = self.config.max_width.checked_sub(pattern_offset.width() + 1);
            let pattern_width = match pattern_width {
                Some(width) => width,
                None => return,
            };

            match local.pat.rewrite(&context, pattern_width, pattern_offset) {
                Some(ref pat_string) => result.push_str(pat_string),
                None => return,
            }

            result.push_str(&infix);

            if let Some(ref ex) = local.init {
                let max_width = self.config.max_width.checked_sub(context.block_indent.width() + 1);
                let max_width = match max_width {
                    Some(width) => width,
                    None => return,
                };

                // 1 = trailing semicolon;
                let rhs = rewrite_assign_rhs(&context, result, ex, max_width, context.block_indent);

                match rhs {
                    Some(result) => result,
                    None => return,
                }
            } else {
                result
            }
        };

        self.buffer.push_str(&result);
        self.buffer.push_str(";");
        self.last_pos = span.hi;
    }

    pub fn format_foreign_mod(&mut self, fm: &ast::ForeignMod, span: Span) {
        self.buffer.push_str("extern ");

        if fm.abi != abi::Abi::C {
            self.buffer.push_str(&format!("{} ", fm.abi));
        }

        let snippet = self.snippet(span);
        let brace_pos = snippet.find_uncommented("{").unwrap() as u32;

        // FIXME: this skips comments between the extern keyword and the opening
        // brace.
        self.last_pos = span.lo + BytePos(brace_pos);
        self.block_indent = self.block_indent.block_indent(self.config);

        for item in &fm.items {
            self.format_foreign_item(&*item);
        }

        self.block_indent = self.block_indent.block_unindent(self.config);
        self.format_missing_with_indent(span.hi - BytePos(1));
        self.buffer.push_str("}");
        self.last_pos = span.hi;
    }

    fn format_foreign_item(&mut self, item: &ast::ForeignItem) {
        self.format_missing_with_indent(item.span.lo);
        // Drop semicolon or it will be interpreted as comment.
        // FIXME: this may be a faulty span from libsyntax.
        let span = codemap::mk_sp(item.span.lo, item.span.hi - BytePos(1));

        match item.node {
            ast::ForeignItem_::ForeignItemFn(ref fn_decl, ref generics) => {
                let indent = self.block_indent;
                let rewrite = self.rewrite_fn_base(indent,
                                                   item.ident,
                                                   fn_decl,
                                                   None,
                                                   generics,
                                                   ast::Unsafety::Normal,
                                                   ast::Constness::NotConst,
                                                   // These are not actually rust functions,
                                                   // but we format them as such.
                                                   abi::Abi::Rust,
                                                   item.vis,
                                                   span,
                                                   false,
                                                   false);

                match rewrite {
                    Some((new_fn, _)) => {
                        self.buffer.push_str(&new_fn);
                        self.buffer.push_str(";");
                    }
                    None => self.format_missing(item.span.hi),
                }
            }
            ast::ForeignItem_::ForeignItemStatic(ref ty, is_mutable) => {
                // FIXME(#21): we're dropping potential comments in between the
                // function keywords here.
                let mut_str = if is_mutable {
                    "mut "
                } else {
                    ""
                };
                let prefix = format!("{}static {}{}: ",
                                     format_visibility(item.vis),
                                     mut_str,
                                     item.ident);
                let offset = self.block_indent + prefix.len();
                // 1 = ;
                let width = self.config.max_width - offset.width() - 1;
                let rewrite = ty.rewrite(&self.get_context(), width, offset);

                match rewrite {
                    Some(result) => {
                        self.buffer.push_str(&prefix);
                        self.buffer.push_str(&result);
                        self.buffer.push_str(";");
                    }
                    None => self.format_missing(item.span.hi),
                }
            }
        }

        self.last_pos = item.span.hi;
    }

    pub fn rewrite_fn(&mut self,
                      indent: Indent,
                      ident: ast::Ident,
                      fd: &ast::FnDecl,
                      explicit_self: Option<&ast::ExplicitSelf>,
                      generics: &ast::Generics,
                      unsafety: ast::Unsafety,
                      constness: ast::Constness,
                      abi: abi::Abi,
                      vis: ast::Visibility,
                      span: Span)
                      -> Option<String> {
        let mut newline_brace = self.newline_for_brace(&generics.where_clause);

        let (mut result, force_newline_brace) = try_opt!(self.rewrite_fn_base(indent,
                                                                              ident,
                                                                              fd,
                                                                              explicit_self,
                                                                              generics,
                                                                              unsafety,
                                                                              constness,
                                                                              abi,
                                                                              vis,
                                                                              span,
                                                                              newline_brace,
                                                                              true));

        if self.config.fn_brace_style != BraceStyle::AlwaysNextLine && !result.contains('\n') {
            newline_brace = false;
        } else if force_newline_brace {
            newline_brace = true;
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

        Some(result)
    }

    pub fn rewrite_required_fn(&mut self,
                               indent: Indent,
                               ident: ast::Ident,
                               sig: &ast::MethodSig,
                               span: Span)
                               -> Option<String> {
        // Drop semicolon or it will be interpreted as comment
        let span = codemap::mk_sp(span.lo, span.hi - BytePos(1));

        // FIXME: silly formatting of the `.0`.
        let mut result = try_opt!(self.rewrite_fn_base(indent,
                                                       ident,
                                                       &sig.decl,
                                                       Some(&sig.explicit_self),
                                                       &sig.generics,
                                                       sig.unsafety,
                                                       sig.constness,
                                                       sig.abi,
                                                       ast::Visibility::Inherited,
                                                       span,
                                                       false,
                                                       false))
                             .0;

        // Re-attach semicolon
        result.push(';');

        Some(result)
    }

    // Return type is (result, force_new_line_for_brace)
    fn rewrite_fn_base(&mut self,
                       indent: Indent,
                       ident: ast::Ident,
                       fd: &ast::FnDecl,
                       explicit_self: Option<&ast::ExplicitSelf>,
                       generics: &ast::Generics,
                       unsafety: ast::Unsafety,
                       constness: ast::Constness,
                       abi: abi::Abi,
                       vis: ast::Visibility,
                       span: Span,
                       newline_brace: bool,
                       has_body: bool)
                       -> Option<(String, bool)> {
        let mut force_new_line_for_brace = false;
        // FIXME we'll lose any comments in between parts of the function decl, but anyone
        // who comments there probably deserves what they get.

        let where_clause = &generics.where_clause;

        let mut result = String::with_capacity(1024);
        // Vis unsafety abi.
        result.push_str(format_visibility(vis));

        if let ast::Unsafety::Unsafe = unsafety {
            result.push_str("unsafe ");
        }
        if let ast::Constness::Const = constness {
            result.push_str("const ");
        }
        if abi != abi::Rust {
            result.push_str("extern ");
            result.push_str(&abi.to_string());
            result.push(' ');
        }

        // fn foo
        result.push_str("fn ");
        result.push_str(&ident.to_string());

        // Generics.
        let generics_indent = indent + result.len();
        let generics_span = codemap::mk_sp(span.lo, span_for_return(&fd.output).lo);
        let generics_str = try_opt!(self.rewrite_generics(generics,
                                                          indent,
                                                          generics_indent,
                                                          generics_span));
        result.push_str(&generics_str);

        let context = self.get_context();
        // Note that if the width and indent really matter, we'll re-layout the
        // return type later anyway.
        let ret_str = fd.output
                        .rewrite(&context, self.config.max_width - indent.width(), indent)
                        .unwrap();

        let multi_line_ret_str = ret_str.contains('\n');
        let ret_str_len = if multi_line_ret_str {
            0
        } else {
            ret_str.len()
        };

        // Args.
        let (mut one_line_budget, multi_line_budget, mut arg_indent) =
            self.compute_budgets_for_args(&result, indent, ret_str_len, newline_brace);

        debug!("rewrite_fn: one_line_budget: {}, multi_line_budget: {}, arg_indent: {:?}",
               one_line_budget,
               multi_line_budget,
               arg_indent);

        // Check if vertical layout was forced by compute_budget_for_args.
        if one_line_budget <= 0 {
            if self.config.fn_args_paren_newline {
                result.push('\n');
                result.push_str(&arg_indent.to_string(self.config));
                arg_indent = arg_indent + 1; // extra space for `(`
                result.push('(');
            } else {
                result.push_str("(\n");
                result.push_str(&arg_indent.to_string(self.config));
            }
        } else if self.config.fn_args_layout == StructLitStyle::Block {
            arg_indent = indent.block_indent(self.config);
            result.push_str("(\n");
            result.push_str(&arg_indent.to_string(self.config));
        } else {
            result.push('(');
        }

        if multi_line_ret_str {
            one_line_budget = 0;
        }

        // A conservative estimation, to goal is to be over all parens in generics
        let args_start = generics.ty_params
                                 .last()
                                 .map(|tp| end_typaram(tp))
                                 .unwrap_or(span.lo);
        let args_span = codemap::mk_sp(span_after(codemap::mk_sp(args_start, span.hi),
                                                  "(",
                                                  self.codemap),
                                       span_for_return(&fd.output).lo);
        let arg_str = try_opt!(self.rewrite_args(&fd.inputs,
                                                 explicit_self,
                                                 one_line_budget,
                                                 multi_line_budget,
                                                 indent,
                                                 arg_indent,
                                                 args_span,
                                                 fd.variadic));
        result.push_str(&arg_str);
        if self.config.fn_args_layout == StructLitStyle::Block {
            result.push('\n');
        }
        result.push(')');

        // Return type.
        if !ret_str.is_empty() {
            // If we've already gone multi-line, or the return type would push
            // over the max width, then put the return type on a new line.
            // Unless we are formatting args like a block, in which case there
            // should always be room for the return type.
            let ret_indent = if (result.contains("\n") || multi_line_ret_str ||
                                 result.len() + indent.width() + ret_str_len >
                                 self.config.max_width) &&
                                self.config.fn_args_layout != StructLitStyle::Block {
                let indent = match self.config.fn_return_indent {
                    ReturnIndent::WithWhereClause => indent + 4,
                    // Aligning with non-existent args looks silly.
                    _ if arg_str.len() == 0 => {
                        force_new_line_for_brace = true;
                        indent + 4
                    }
                    // FIXME: we might want to check that using the arg indent
                    // doesn't blow our budget, and if it does, then fallback to
                    // the where clause indent.
                    _ => arg_indent,
                };

                result.push('\n');
                result.push_str(&indent.to_string(self.config));
                indent
            } else {
                result.push(' ');
                Indent::new(indent.width(), result.len())
            };

            if multi_line_ret_str {
                // Now that we know the proper indent and width, we need to
                // re-layout the return type.

                let budget = try_opt!(self.config.max_width.checked_sub(ret_indent.width()));
                let ret_str = fd.output
                                .rewrite(&context, budget, ret_indent)
                                .unwrap();
                result.push_str(&ret_str);
            } else {
                result.push_str(&ret_str);
            }

            // Comment between return type and the end of the decl.
            let snippet_lo = fd.output.span().hi;
            if where_clause.predicates.is_empty() {
                let snippet_hi = span.hi;
                let snippet = self.snippet(codemap::mk_sp(snippet_lo, snippet_hi));
                let snippet = snippet.trim();
                if !snippet.is_empty() {
                    result.push(' ');
                    result.push_str(snippet);
                }
            } else {
                // FIXME it would be nice to catch comments between the return type
                // and the where clause, but we don't have a span for the where
                // clause.
            }
        }

        let where_density = if (self.config.where_density == Density::Compressed &&
                                (!result.contains('\n') ||
                                 self.config.fn_args_layout == StructLitStyle::Block)) ||
                               (self.config.fn_args_layout == StructLitStyle::Block &&
                                ret_str.is_empty()) ||
                               (self.config.where_density == Density::CompressedIfEmpty &&
                                !has_body) {
            Density::Compressed
        } else {
            Density::Tall
        };

        // Where clause.
        let where_clause_str = try_opt!(self.rewrite_where_clause(where_clause,
                                                                  self.config,
                                                                  indent,
                                                                  where_density,
                                                                  span.hi));
        result.push_str(&where_clause_str);

        Some((result, force_new_line_for_brace))
    }

    fn rewrite_args(&self,
                    args: &[ast::Arg],
                    explicit_self: Option<&ast::ExplicitSelf>,
                    one_line_budget: usize,
                    multi_line_budget: usize,
                    indent: Indent,
                    arg_indent: Indent,
                    span: Span,
                    variadic: bool)
                    -> Option<String> {
        let context = self.get_context();
        let mut arg_item_strs = try_opt!(args.iter()
                                             .map(|arg| {
                                                 arg.rewrite(&context,
                                                             multi_line_budget,
                                                             arg_indent)
                                             })
                                             .collect::<Option<Vec<_>>>());

        // Account for sugary self.
        // FIXME: the comment for the self argument is dropped. This is blocked
        // on rust issue #27522.
        let min_args = explicit_self.and_then(|explicit_self| {
                                        rewrite_explicit_self(explicit_self, args)
                                    })
                                    .map(|self_str| {
                                        arg_item_strs[0] = self_str;
                                        2
                                    })
                                    .unwrap_or(1);

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
                span_after(span, ",", self.codemap)
            } else {
                span.lo
            };

            enum ArgumentKind<'a> {
                Regular(&'a ast::Arg),
                Variadic(BytePos),
            }

            let variadic_arg = if variadic {
                let variadic_span = codemap::mk_sp(args.last().unwrap().ty.span.hi, span.hi);
                let variadic_start = span_after(variadic_span, "...", self.codemap) - BytePos(3);
                Some(ArgumentKind::Variadic(variadic_start))
            } else {
                None
            };

            let more_items = itemize_list(self.codemap,
                                          args[min_args - 1..]
                                              .iter()
                                              .map(ArgumentKind::Regular)
                                              .chain(variadic_arg),
                                          ")",
                                          |arg| {
                                              match *arg {
                                                  ArgumentKind::Regular(arg) =>
                                                      span_lo_for_arg(arg),
                                                  ArgumentKind::Variadic(start) => start,
                                              }
                                          },
                                          |arg| {
                                              match *arg {
                                                  ArgumentKind::Regular(arg) => arg.ty.span.hi,
                                                  ArgumentKind::Variadic(start) =>
                                                      start + BytePos(3),
                                              }
                                          },
                                          |arg| {
                                              match *arg {
                                                  ArgumentKind::Regular(..) => None,
                                                  ArgumentKind::Variadic(..) =>
                                                      Some("...".to_owned()),
                                              }
                                          },
                                          comment_span_start,
                                          span.hi);

            arg_items.extend(more_items);
        }

        for (item, arg) in arg_items.iter_mut().zip(arg_item_strs) {
            item.item = Some(arg);
        }

        let indent = match self.config.fn_arg_indent {
            BlockIndentStyle::Inherit => indent,
            BlockIndentStyle::Tabbed => indent.block_indent(self.config),
            BlockIndentStyle::Visual => arg_indent,
        };

        let tactic = definitive_tactic(&arg_items,
                                       self.config.fn_args_density.to_list_tactic(),
                                       one_line_budget);
        let budget = match tactic {
            DefinitiveListTactic::Horizontal => one_line_budget,
            _ => multi_line_budget,
        };

        let fmt = ListFormatting {
            tactic: tactic,
            separator: ",",
            trailing_separator: SeparatorTactic::Never,
            indent: indent,
            width: budget,
            ends_with_newline: false,
            config: self.config,
        };

        write_list(&arg_items, &fmt)
    }

    fn compute_budgets_for_args(&self,
                                result: &str,
                                indent: Indent,
                                ret_str_len: usize,
                                newline_brace: bool)
                                -> (usize, usize, Indent) {
        // Try keeping everything on the same line
        if !result.contains("\n") {
            // 3 = `() `, space is before ret_string
            let mut used_space = indent.width() + result.len() + ret_str_len + 3;
            if !newline_brace {
                used_space += 2;
            }
            let one_line_budget = if used_space > self.config.max_width {
                0
            } else {
                self.config.max_width - used_space
            };

            // 2 = `()`
            let used_space = indent.width() + result.len() + 2;
            let max_space = self.config.max_width;
            debug!("compute_budgets_for_args: used_space: {}, max_space: {}",
                   used_space,
                   max_space);
            if used_space < max_space {
                return (one_line_budget,
                        max_space - used_space,
                        indent + result.len() + 1);
            }
        }

        // Didn't work. we must force vertical layout and put args on a newline.
        let new_indent = indent.block_indent(self.config);
        let used_space = new_indent.width() + 2; // account for `(` and `)`
        let max_space = self.config.max_width;
        if used_space <= max_space {
            (0, max_space - used_space, new_indent)
        } else {
            // Whoops! bankrupt.
            // FIXME: take evasive action, perhaps kill the indent or something.
            panic!("in compute_budgets_for_args");
        }
    }

    fn newline_for_brace(&self, where_clause: &ast::WhereClause) -> bool {
        match self.config.fn_brace_style {
            BraceStyle::AlwaysNextLine => true,
            BraceStyle::SameLineWhere if !where_clause.predicates.is_empty() => true,
            _ => false,
        }
    }

    pub fn visit_enum(&mut self,
                      ident: ast::Ident,
                      vis: ast::Visibility,
                      enum_def: &ast::EnumDef,
                      generics: &ast::Generics,
                      span: Span) {
        let header_str = self.format_header("enum ", ident, vis);
        self.buffer.push_str(&header_str);

        let enum_snippet = self.snippet(span);
        let body_start = span.lo + BytePos(enum_snippet.find_uncommented("{").unwrap() as u32 + 1);
        let generics_str = self.format_generics(generics,
                                                " {",
                                                self.block_indent,
                                                self.block_indent.block_indent(self.config),
                                                codemap::mk_sp(span.lo, body_start))
                               .unwrap();
        self.buffer.push_str(&generics_str);

        self.last_pos = body_start;

        self.block_indent = self.block_indent.block_indent(self.config);
        let variant_list = self.format_variant_list(enum_def, body_start, span.hi - BytePos(1));
        match variant_list {
            Some(ref body_str) => self.buffer.push_str(&body_str),
            None => self.format_missing(span.hi - BytePos(1)),
        }
        self.block_indent = self.block_indent.block_unindent(self.config);

        if variant_list.is_some() {
            self.buffer.push_str(&self.block_indent.to_string(self.config));
        }
        self.buffer.push_str("}");
        self.last_pos = span.hi;
    }

    // Format the body of an enum definition
    fn format_variant_list(&self,
                           enum_def: &ast::EnumDef,
                           body_lo: BytePos,
                           body_hi: BytePos)
                           -> Option<String> {
        if enum_def.variants.is_empty() {
            return None;
        }
        let mut result = String::with_capacity(1024);
        result.push('\n');
        let indentation = self.block_indent.to_string(self.config);
        result.push_str(&indentation);

        let items = itemize_list(self.codemap,
                                 enum_def.variants.iter(),
                                 "}",
                                 |f| {
                                     if !f.node.attrs.is_empty() {
                                         f.node.attrs[0].span.lo
                                     } else {
                                         f.span.lo
                                     }
                                 },
                                 |f| f.span.hi,
                                 |f| self.format_variant(f),
                                 body_lo,
                                 body_hi);

        let budget = self.config.max_width - self.block_indent.width() - 2;
        let fmt = ListFormatting {
            tactic: DefinitiveListTactic::Vertical,
            separator: ",",
            trailing_separator: SeparatorTactic::Always,
            indent: self.block_indent,
            width: budget,
            ends_with_newline: true,
            config: self.config,
        };

        let list = try_opt!(write_list(items, &fmt));
        result.push_str(&list);
        result.push('\n');
        Some(result)
    }

    // Variant of an enum.
    fn format_variant(&self, field: &ast::Variant) -> Option<String> {
        if contains_skip(&field.node.attrs) {
            let lo = field.node.attrs[0].span.lo;
            let span = codemap::mk_sp(lo, field.span.hi);
            return Some(self.snippet(span));
        }

        let indent = self.block_indent;
        let mut result = try_opt!(field.node
                                       .attrs
                                       .rewrite(&self.get_context(),
                                                self.config.max_width - indent.width(),
                                                indent));
        if !result.is_empty() {
            result.push('\n');
            result.push_str(&indent.to_string(self.config));
        }

        let variant_body = match *field.node.data {
            ast::VariantData::Tuple(ref types, _) => {
                let mut result = field.node.name.to_string();
                let items = itemize_list(self.codemap,
                                         types.iter(),
                                         ")",
                                         |arg| arg.node.ty.span.lo,
                                         |arg| arg.node.ty.span.hi,
                                         |arg| {
                                             // FIXME silly width, indent
                                             arg.node
                                                .ty
                                                .rewrite(&self.get_context(), 1000, Indent::empty())
                                         },
                                         span_after(field.span, "(", self.codemap),
                                         field.span.hi);
                let item_vec = items.collect::<Vec<_>>();

                result.push('(');

                let indent = indent + field.node.name.to_string().len() + "(".len();

                let comma_cost = if self.config.enum_trailing_comma {
                    1
                } else {
                    0
                };
                let budget = self.config.max_width - indent.width() - comma_cost - 1; // 1 = )
                let tactic = definitive_tactic(&item_vec, ListTactic::HorizontalVertical, budget);

                let fmt = ListFormatting {
                    tactic: tactic,
                    separator: ",",
                    trailing_separator: SeparatorTactic::Never,
                    indent: indent,
                    width: budget,
                    ends_with_newline: true,
                    config: self.config,
                };
                let list_str = try_opt!(write_list(&item_vec, &fmt));

                result.push_str(&list_str);
                result.push(')');

                Some(result)
            }
            ast::VariantData::Struct(..) => {
                // FIXME: Should limit the width, as we have a trailing comma
                self.format_struct("",
                                   field.node.name,
                                   ast::Visibility::Inherited,
                                   &*field.node.data,
                                   None,
                                   field.span,
                                   indent)
            }
            ast::VariantData::Unit(..) => {
                let tag = if let Some(ref expr) = field.node.disr_expr {
                    format!("{} = {}", field.node.name, self.snippet(expr.span))
                } else {
                    field.node.name.to_string()
                };

                wrap_str(tag,
                         self.config.max_width,
                         self.config.max_width - indent.width(),
                         indent)
            }
        };

        if let Some(variant_str) = variant_body {
            result.push_str(&variant_str);
            Some(result)
        } else {
            None
        }
    }

    pub fn format_struct(&self,
                         item_name: &str,
                         ident: ast::Ident,
                         vis: ast::Visibility,
                         struct_def: &ast::VariantData,
                         generics: Option<&ast::Generics>,
                         span: Span,
                         offset: Indent)
                         -> Option<String> {
        let mut result = String::with_capacity(1024);

        let header_str = self.format_header(item_name, ident, vis);
        result.push_str(&header_str);

        let (is_tuple, fields) = match *struct_def {
            ast::VariantData::Unit(..) => {
                result.push(';');
                return Some(result);
            }
            ast::VariantData::Tuple(ref vec, _) => (true, vec),
            ast::VariantData::Struct(ref vec, _) => (false, vec),
        };

        let (opener, terminator) = if is_tuple {
            ("(", ")")
        } else {
            (" {", "}")
        };

        let generics_str = match generics {
            Some(g) => {
                try_opt!(self.format_generics(g,
                                              opener,
                                              offset,
                                              offset + header_str.len(),
                                              codemap::mk_sp(span.lo, fields[0].span.lo)))
            }
            None => opener.to_owned(),
        };
        result.push_str(&generics_str);

        let items = itemize_list(self.codemap,
                                 fields.iter(),
                                 terminator,
                                 |field| {
                                     // Include attributes and doc comments, if present
                                     if !field.node.attrs.is_empty() {
                                         field.node.attrs[0].span.lo
                                     } else {
                                         field.span.lo
                                     }
                                 },
                                 |field| field.node.ty.span.hi,
                                 |field| self.format_field(field),
                                 span_after(span, opener.trim(), self.codemap),
                                 span.hi);

        // 2 terminators and a semicolon
        let used_budget = offset.width() + header_str.len() + generics_str.len() + 3;

        // Conservative approximation
        let single_line_cost = (span.hi - fields[0].span.lo).0;
        let break_line = !is_tuple || generics_str.contains('\n') ||
                         single_line_cost as usize + used_budget > self.config.max_width;

        let tactic = if break_line {
            let indentation = offset.block_indent(self.config).to_string(self.config);
            result.push('\n');
            result.push_str(&indentation);

            DefinitiveListTactic::Vertical
        } else {
            DefinitiveListTactic::Horizontal
        };

        // 1 = ,
        let budget = self.config.max_width - offset.width() + self.config.tab_spaces - 1;
        let fmt = ListFormatting {
            tactic: tactic,
            separator: ",",
            trailing_separator: self.config.struct_trailing_comma,
            indent: offset.block_indent(self.config),
            width: budget,
            ends_with_newline: true,
            config: self.config,
        };

        let list_str = try_opt!(write_list(items, &fmt));
        result.push_str(&list_str);

        if break_line {
            result.push('\n');
            result.push_str(&offset.to_string(self.config));
        }

        result.push_str(terminator);

        if is_tuple {
            result.push(';');
        }

        Some(result)
    }

    fn format_header(&self, item_name: &str, ident: ast::Ident, vis: ast::Visibility) -> String {
        format!("{}{}{}", format_visibility(vis), item_name, ident)
    }

    fn format_generics(&self,
                       generics: &ast::Generics,
                       opener: &str,
                       offset: Indent,
                       generics_offset: Indent,
                       span: Span)
                       -> Option<String> {
        let mut result = try_opt!(self.rewrite_generics(generics, offset, generics_offset, span));

        if !generics.where_clause.predicates.is_empty() || result.contains('\n') {
            let where_clause_str = try_opt!(self.rewrite_where_clause(&generics.where_clause,
                                                                      self.config,
                                                                      self.block_indent,
                                                                      Density::Tall,
                                                                      span.hi));
            result.push_str(&where_clause_str);
            result.push_str(&self.block_indent.to_string(self.config));
            result.push('\n');
            result.push_str(opener.trim());
        } else {
            result.push_str(opener);
        }

        Some(result)
    }

    // Field of a struct
    fn format_field(&self, field: &ast::StructField) -> Option<String> {
        if contains_skip(&field.node.attrs) {
            // FIXME: silly width, indent
            return wrap_str(self.snippet(codemap::mk_sp(field.node.attrs[0].span.lo,
                                                        field.span.hi)),
                            self.config.max_width,
                            1000,
                            Indent::empty());
        }

        let name = match field.node.kind {
            ast::StructFieldKind::NamedField(ident, _) => Some(ident.to_string()),
            ast::StructFieldKind::UnnamedField(_) => None,
        };
        let vis = match field.node.kind {
            ast::StructFieldKind::NamedField(_, vis) |
            ast::StructFieldKind::UnnamedField(vis) => format_visibility(vis),
        };
        // FIXME silly width, indent
        let typ = try_opt!(field.node.ty.rewrite(&self.get_context(), 1000, Indent::empty()));

        let indent = self.block_indent.block_indent(self.config);
        let mut attr_str = try_opt!(field.node
                                         .attrs
                                         .rewrite(&self.get_context(),
                                                  self.config.max_width - indent.width(),
                                                  indent));
        if !attr_str.is_empty() {
            attr_str.push('\n');
            attr_str.push_str(&indent.to_string(self.config));
        }

        Some(match name {
            Some(name) => format!("{}{}{}: {}", attr_str, vis, name, typ),
            None => format!("{}{}{}", attr_str, vis, typ),
        })
    }

    fn rewrite_generics(&self,
                        generics: &ast::Generics,
                        offset: Indent,
                        generics_offset: Indent,
                        span: Span)
                        -> Option<String> {
        // FIXME: convert bounds to where clauses where they get too big or if
        // there is a where clause at all.
        let lifetimes: &[_] = &generics.lifetimes;
        let tys: &[_] = &generics.ty_params;
        if lifetimes.is_empty() && tys.is_empty() {
            return Some(String::new());
        }

        let offset = match self.config.generics_indent {
            BlockIndentStyle::Inherit => offset,
            BlockIndentStyle::Tabbed => offset.block_indent(self.config),
            // 1 = <
            BlockIndentStyle::Visual => generics_offset + 1,
        };

        let h_budget = self.config.max_width - generics_offset.width() - 2;
        // FIXME: might need to insert a newline if the generics are really long.

        // Strings for the generics.
        let context = self.get_context();
        let lt_strs = lifetimes.iter().map(|lt| lt.rewrite(&context, h_budget, offset));
        let ty_strs = tys.iter().map(|ty_param| ty_param.rewrite(&context, h_budget, offset));

        // Extract comments between generics.
        let lt_spans = lifetimes.iter().map(|l| {
            let hi = if l.bounds.is_empty() {
                l.lifetime.span.hi
            } else {
                l.bounds[l.bounds.len() - 1].span.hi
            };
            codemap::mk_sp(l.lifetime.span.lo, hi)
        });
        let ty_spans = tys.iter().map(span_for_ty_param);

        let items = itemize_list(self.codemap,
                                 lt_spans.chain(ty_spans).zip(lt_strs.chain(ty_strs)),
                                 ">",
                                 |&(sp, _)| sp.lo,
                                 |&(sp, _)| sp.hi,
                                 // FIXME: don't clone
                                 |&(_, ref str)| str.clone(),
                                 span_after(span, "<", self.codemap),
                                 span.hi);
        let list_str = try_opt!(::lists::format_item_list(items, h_budget, offset, self.config));

        Some(format!("<{}>", list_str))
    }

    fn rewrite_where_clause(&self,
                            where_clause: &ast::WhereClause,
                            config: &Config,
                            indent: Indent,
                            density: Density,
                            span_end: BytePos)
                            -> Option<String> {
        if where_clause.predicates.is_empty() {
            return Some(String::new());
        }

        let extra_indent = match self.config.where_indent {
            BlockIndentStyle::Inherit => Indent::empty(),
            BlockIndentStyle::Tabbed | BlockIndentStyle::Visual =>
                Indent::new(config.tab_spaces, 0),
        };

        let context = self.get_context();

        let offset = match self.config.where_pred_indent {
            BlockIndentStyle::Inherit => indent + extra_indent,
            BlockIndentStyle::Tabbed => indent + extra_indent.block_indent(config),
            // 6 = "where ".len()
            BlockIndentStyle::Visual => indent + extra_indent + 6,
        };
        // FIXME: if where_pred_indent != Visual, then the budgets below might
        // be out by a char or two.

        let budget = self.config.max_width - offset.width();
        let span_start = span_for_where_pred(&where_clause.predicates[0]).lo;
        let items = itemize_list(self.codemap,
                                 where_clause.predicates.iter(),
                                 "{",
                                 |pred| span_for_where_pred(pred).lo,
                                 |pred| span_for_where_pred(pred).hi,
                                 |pred| pred.rewrite(&context, budget, offset),
                                 span_start,
                                 span_end);
        let item_vec = items.collect::<Vec<_>>();
        // FIXME: we don't need to collect here if the where_layout isnt horizontalVertical
        let tactic = definitive_tactic(&item_vec, self.config.where_layout, budget);

        let fmt = ListFormatting {
            tactic: tactic,
            separator: ",",
            trailing_separator: SeparatorTactic::Never,
            indent: offset,
            width: budget,
            ends_with_newline: true,
            config: self.config,
        };
        let preds_str = try_opt!(write_list(&item_vec, &fmt));

        // 9 = " where ".len() + " {".len()
        if density == Density::Tall || preds_str.contains('\n') ||
           indent.width() + 9 + preds_str.len() > self.config.max_width {
            Some(format!("\n{}where {}",
                         (indent + extra_indent).to_string(self.config),
                         preds_str))
        } else {
            Some(format!(" where {}", preds_str))
        }
    }
}

pub fn rewrite_static(prefix: &str,
                      vis: ast::Visibility,
                      ident: ast::Ident,
                      ty: &ast::Ty,
                      mutability: ast::Mutability,
                      expr: &ast::Expr,
                      context: &RewriteContext)
                      -> Option<String> {
    let prefix = format!("{}{} {}{}: ",
                         format_visibility(vis),
                         prefix,
                         format_mutability(mutability),
                         ident);
    // 2 = " =".len()
    let ty_str = try_opt!(ty.rewrite(context,
                                     context.config.max_width - context.block_indent.width() -
                                     prefix.len() - 2,
                                     context.block_indent));
    let lhs = format!("{}{} =", prefix, ty_str);

    // 1 = ;
    let remaining_width = context.config.max_width - context.block_indent.width() - 1;
    rewrite_assign_rhs(context, lhs, expr, remaining_width, context.block_indent).map(|s| s + ";")
}

impl Rewrite for ast::FunctionRetTy {
    fn rewrite(&self, context: &RewriteContext, width: usize, offset: Indent) -> Option<String> {
        match *self {
            ast::FunctionRetTy::DefaultReturn(_) => Some(String::new()),
            ast::FunctionRetTy::NoReturn(_) => {
                if width >= 4 {
                    Some("-> !".to_owned())
                } else {
                    None
                }
            }
            ast::FunctionRetTy::Return(ref ty) => {
                let inner_width = try_opt!(width.checked_sub(3));
                ty.rewrite(context, inner_width, offset + 3).map(|r| format!("-> {}", r))
            }
        }
    }
}

impl Rewrite for ast::Arg {
    fn rewrite(&self, context: &RewriteContext, width: usize, offset: Indent) -> Option<String> {
        if is_named_arg(self) {
            if let ast::Ty_::TyInfer = self.ty.node {
                wrap_str(pprust::pat_to_string(&self.pat),
                         context.config.max_width,
                         width,
                         offset)
            } else {
                let mut result = pprust::pat_to_string(&self.pat);
                result.push_str(": ");
                let max_width = try_opt!(width.checked_sub(result.len()));
                let ty_str = try_opt!(self.ty.rewrite(context, max_width, offset + result.len()));
                result.push_str(&ty_str);
                Some(result)
            }
        } else {
            self.ty.rewrite(context, width, offset)
        }
    }
}

fn rewrite_explicit_self(explicit_self: &ast::ExplicitSelf, args: &[ast::Arg]) -> Option<String> {
    match explicit_self.node {
        ast::ExplicitSelf_::SelfRegion(lt, m, _) => {
            let mut_str = format_mutability(m);
            match lt {
                Some(ref l) => Some(format!("&{} {}self", pprust::lifetime_to_string(l), mut_str)),
                None => Some(format!("&{}self", mut_str)),
            }
        }
        ast::ExplicitSelf_::SelfExplicit(ref ty, _) => {
            Some(format!("self: {}", pprust::ty_to_string(ty)))
        }
        ast::ExplicitSelf_::SelfValue(_) => {
            assert!(args.len() >= 1, "&[ast::Arg] shouldn't be empty.");

            // this hacky solution caused by absence of `Mutability` in `SelfValue`.
            let mut_str = {
                if let ast::Pat_::PatIdent(ast::BindingMode::BindByValue(mutability), _, _) =
                       args[0].pat.node {
                    format_mutability(mutability)
                } else {
                    panic!("there is a bug or change in structure of AST, aborting.");
                }
            };

            Some(format!("{}self", mut_str))
        }
        _ => None,
    }
}

pub fn span_lo_for_arg(arg: &ast::Arg) -> BytePos {
    if is_named_arg(arg) {
        arg.pat.span.lo
    } else {
        arg.ty.span.lo
    }
}

pub fn span_hi_for_arg(arg: &ast::Arg) -> BytePos {
    match arg.ty.node {
        ast::Ty_::TyInfer if is_named_arg(arg) => arg.pat.span.hi,
        _ => arg.ty.span.hi,
    }
}

fn is_named_arg(arg: &ast::Arg) -> bool {
    if let ast::Pat_::PatIdent(_, ident, _) = arg.pat.node {
        ident.node != token::special_idents::invalid
    } else {
        true
    }
}

fn span_for_return(ret: &ast::FunctionRetTy) -> Span {
    match *ret {
        ast::FunctionRetTy::NoReturn(ref span) |
        ast::FunctionRetTy::DefaultReturn(ref span) => span.clone(),
        ast::FunctionRetTy::Return(ref ty) => ty.span,
    }
}

fn span_for_ty_param(ty: &ast::TyParam) -> Span {
    // Note that ty.span is the span for ty.ident, not the whole item.
    let lo = ty.span.lo;
    if let Some(ref def) = ty.default {
        return codemap::mk_sp(lo, def.span.hi);
    }
    if ty.bounds.is_empty() {
        return ty.span;
    }
    let hi = match ty.bounds[ty.bounds.len() - 1] {
        ast::TyParamBound::TraitTyParamBound(ref ptr, _) => ptr.span.hi,
        ast::TyParamBound::RegionTyParamBound(ref l) => l.span.hi,
    };
    codemap::mk_sp(lo, hi)
}

fn span_for_where_pred(pred: &ast::WherePredicate) -> Span {
    match *pred {
        ast::WherePredicate::BoundPredicate(ref p) => p.span,
        ast::WherePredicate::RegionPredicate(ref p) => p.span,
        ast::WherePredicate::EqPredicate(ref p) => p.span,
    }
}
