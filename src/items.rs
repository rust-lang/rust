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

use {ReturnIndent, BraceStyle};
use utils::{format_visibility, make_indent, contains_skip, span_after, end_typaram};
use lists::{write_list, itemize_list, ListItem, ListFormatting, SeparatorTactic, ListTactic};
use comment::FindUncommented;
use visitor::FmtVisitor;

use syntax::{ast, abi};
use syntax::codemap::{self, Span, BytePos};
use syntax::print::pprust;
use syntax::parse::token;

impl<'a> FmtVisitor<'a> {
    pub fn rewrite_fn(&mut self,
                      indent: usize,
                      ident: ast::Ident,
                      fd: &ast::FnDecl,
                      explicit_self: Option<&ast::ExplicitSelf>,
                      generics: &ast::Generics,
                      unsafety: &ast::Unsafety,
                      constness: &ast::Constness,
                      abi: &abi::Abi,
                      vis: ast::Visibility,
                      span: Span)
                      -> String {
        let newline_brace = self.newline_for_brace(&generics.where_clause);

        let mut result = self.rewrite_fn_base(indent,
                                              ident,
                                              fd,
                                              explicit_self,
                                              generics,
                                              unsafety,
                                              constness,
                                              abi,
                                              vis,
                                              span,
                                              newline_brace);

        // Prepare for the function body by possibly adding a newline and indent.
        // FIXME we'll miss anything between the end of the signature and the start
        // of the body, but we need more spans from the compiler to solve this.
        if newline_brace {
            result.push('\n');
            result.push_str(&make_indent(indent));
        } else {
            result.push(' ');
        }

        result
    }

    pub fn rewrite_required_fn(&mut self,
                               indent: usize,
                               ident: ast::Ident,
                               sig: &ast::MethodSig,
                               span: Span)
                               -> String {
        // Drop semicolon or it will be interpreted as comment
        let span = codemap::mk_sp(span.lo, span.hi - BytePos(1));

        let mut result = self.rewrite_fn_base(indent,
                                              ident,
                                              &sig.decl,
                                              Some(&sig.explicit_self),
                                              &sig.generics,
                                              &sig.unsafety,
                                              &sig.constness,
                                              &sig.abi,
                                              ast::Visibility::Inherited,
                                              span,
                                              false);

        // Re-attach semicolon
        result.push(';');

        result
    }

    fn rewrite_fn_base(&mut self,
                       indent: usize,
                       ident: ast::Ident,
                       fd: &ast::FnDecl,
                       explicit_self: Option<&ast::ExplicitSelf>,
                       generics: &ast::Generics,
                       unsafety: &ast::Unsafety,
                       constness: &ast::Constness,
                       abi: &abi::Abi,
                       vis: ast::Visibility,
                       span: Span,
                       newline_brace: bool)
                       -> String {
        // FIXME we'll lose any comments in between parts of the function decl, but anyone
        // who comments there probably deserves what they get.

        let where_clause = &generics.where_clause;

        let mut result = String::with_capacity(1024);
        // Vis unsafety abi.
        result.push_str(format_visibility(vis));

        if let &ast::Unsafety::Unsafe = unsafety {
            result.push_str("unsafe ");
        }
        if let &ast::Constness::Const = constness {
            result.push_str("const ");
        }
        if *abi != abi::Rust {
            result.push_str("extern ");
            result.push_str(&abi.to_string());
            result.push(' ');
        }

        // fn foo
        result.push_str("fn ");
        result.push_str(&token::get_ident(ident));

        // Generics.
        let generics_indent = indent + result.len();
        result.push_str(&self.rewrite_generics(generics,
                                               generics_indent,
                                               codemap::mk_sp(span.lo,
                                                              span_for_return(&fd.output).lo)));

        let ret_str = self.rewrite_return(&fd.output);

        // Args.
        let (one_line_budget, multi_line_budget, mut arg_indent) =
            self.compute_budgets_for_args(&result, indent, ret_str.len(), newline_brace);

        debug!("rewrite_fn: one_line_budget: {}, multi_line_budget: {}, arg_indent: {}",
               one_line_budget, multi_line_budget, arg_indent);

        // Check if vertical layout was forced by compute_budget_for_args.
        if one_line_budget <= 0 {
            if self.config.fn_args_paren_newline {
                result.push('\n');
                result.push_str(&make_indent(arg_indent));
                arg_indent = arg_indent + 1; // extra space for `(`
                result.push('(');
            } else {
                result.push_str("(\n");
                result.push_str(&make_indent(arg_indent));
            }
        } else {
            result.push('(');
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
        result.push_str(&self.rewrite_args(&fd.inputs,
                                           explicit_self,
                                           one_line_budget,
                                           multi_line_budget,
                                           arg_indent,
                                           args_span));
        result.push(')');

        // Return type.
        if ret_str.len() > 0 {
            // If we've already gone multi-line, or the return type would push
            // over the max width, then put the return type on a new line.
            if result.contains("\n") ||
               result.len() + indent + ret_str.len() > self.config.max_width {
                let indent = match self.config.fn_return_indent {
                    ReturnIndent::WithWhereClause => indent + 4,
                    // TODO we might want to check that using the arg indent doesn't
                    // blow our budget, and if it does, then fallback to the where
                    // clause indent.
                    _ => arg_indent,
                };

                result.push('\n');
                result.push_str(&make_indent(indent));
            } else {
                result.push(' ');
            }
            result.push_str(&ret_str);

            // Comment between return type and the end of the decl.
            let snippet_lo = fd.output.span().hi;
            if where_clause.predicates.len() == 0 {
                let snippet_hi = span.hi;
                let snippet = self.snippet(codemap::mk_sp(snippet_lo, snippet_hi));
                let snippet = snippet.trim();
                if snippet.len() > 0 {
                    result.push(' ');
                    result.push_str(snippet);
                }
            } else {
                // FIXME it would be nice to catch comments between the return type
                // and the where clause, but we don't have a span for the where
                // clause.
            }
        }

        // Where clause.
        result.push_str(&self.rewrite_where_clause(where_clause,
                                                   indent,
                                                   span.hi));

        result
    }

    fn rewrite_args(&self,
                    args: &[ast::Arg],
                    explicit_self: Option<&ast::ExplicitSelf>,
                    one_line_budget: usize,
                    multi_line_budget: usize,
                    arg_indent: usize,
                    span: Span)
                    -> String {
        let mut arg_item_strs: Vec<_> = args.iter().map(|a| self.rewrite_fn_input(a)).collect();
        // Account for sugary self.
        let mut min_args = 1;
        if let Some(explicit_self) = explicit_self {
            match explicit_self.node {
                ast::ExplicitSelf_::SelfRegion(ref lt, ref m, _) => {
                    let lt_str = match lt {
                        &Some(ref l) => format!("{} ", pprust::lifetime_to_string(l)),
                        &None => String::new(),
                    };
                    let mut_str = match m {
                        &ast::Mutability::MutMutable => "mut ".to_owned(),
                        &ast::Mutability::MutImmutable => String::new(),
                    };
                    arg_item_strs[0] = format!("&{}{}self", lt_str, mut_str);
                    min_args = 2;
                }
                ast::ExplicitSelf_::SelfExplicit(ref ty, _) => {
                    arg_item_strs[0] = format!("self: {}", pprust::ty_to_string(ty));
                }
                ast::ExplicitSelf_::SelfValue(_) => {
                    assert!(args.len() >= 1, "&[ast::Arg] shouldn't be empty.");

                    // this hacky solution caused by absence of `Mutability` in `SelfValue`.
                    let mut_str = {
                        if let ast::Pat_::PatIdent(ast::BindingMode::BindByValue(mutability), _, _)
                                = args[0].pat.node {
                            match mutability {
                                ast::Mutability::MutMutable => "mut ",
                                ast::Mutability::MutImmutable => "",
                            }
                        } else {
                            panic!("there is a bug or change in structure of AST, aborting.");
                        }
                    };

                    arg_item_strs[0] = format!("{}self", mut_str);
                    min_args = 2;
                }
                _ => {}
            }
        }

        // Comments between args
        let mut arg_items = Vec::new();
        if min_args == 2 {
            arg_items.push(ListItem::from_str(""));
        }

        // TODO if there are no args, there might still be a comment, but without
        // spans for the comment or parens, there is no chance of getting it right.
        // You also don't get to put a comment on self, unless it is explicit.
        if args.len() >= min_args {
            let comment_span_start = if min_args == 2 {
                span_after(span, ",", self.codemap)
            } else {
                span.lo
            };

            arg_items = itemize_list(self.codemap,
                                     arg_items,
                                     args[min_args-1..].iter(),
                                     ",",
                                     ")",
                                     |arg| arg.pat.span.lo,
                                     |arg| arg.ty.span.hi,
                                     |_| String::new(),
                                     comment_span_start,
                                     span.hi);
        }

        assert_eq!(arg_item_strs.len(), arg_items.len());

        for (item, arg) in arg_items.iter_mut().zip(arg_item_strs) {
            item.item = arg;
        }

        let fmt = ListFormatting { tactic: ListTactic::HorizontalVertical,
                                   separator: ",",
                                   trailing_separator: SeparatorTactic::Never,
                                   indent: arg_indent,
                                   h_width: one_line_budget,
                                   v_width: multi_line_budget,
                                   ends_with_newline: true, };

        write_list(&arg_items, &fmt)
    }

    fn compute_budgets_for_args(&self,
                                result: &str,
                                indent: usize,
                                ret_str_len: usize,
                                newline_brace: bool)
                                -> (usize, usize, usize) {
        let mut budgets = None;

        // Try keeping everything on the same line
        if !result.contains("\n") {
            // 3 = `() `, space is before ret_string
            let mut used_space = indent + result.len() + ret_str_len + 3;
            if !newline_brace {
                used_space += 2;
            }
            let one_line_budget = if used_space > self.config.max_width {
                0
            } else {
                self.config.max_width - used_space
            };

            // 2 = `()`
            let used_space = indent + result.len() + 2;
            let max_space = self.config.ideal_width + self.config.leeway;
            debug!("compute_budgets_for_args: used_space: {}, max_space: {}",
                   used_space, max_space);
            if used_space < max_space {
                budgets = Some((one_line_budget,
                                max_space - used_space,
                                indent + result.len() + 1));
            }
        }

        // Didn't work. we must force vertical layout and put args on a newline.
        if let None = budgets {
            let new_indent = indent + self.config.tab_spaces;
            let used_space = new_indent + 2; // account for `(` and `)`
            let max_space = self.config.ideal_width + self.config.leeway;
            if used_space > max_space {
                // Whoops! bankrupt.
                // TODO take evasive action, perhaps kill the indent or something.
            } else {
                budgets = Some((0, max_space - used_space, new_indent));
            }
        }

        budgets.unwrap()
    }

    fn newline_for_brace(&self, where_clause: &ast::WhereClause) -> bool {
        match self.config.fn_brace_style {
            BraceStyle::AlwaysNextLine => true,
            BraceStyle::SameLineWhere if where_clause.predicates.len() > 0 => true,
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
        self.changes.push_str_span(span, &header_str);

        let enum_snippet = self.snippet(span);
        let body_start = span.lo + BytePos(enum_snippet.find_uncommented("{").unwrap() as u32 + 1);
        let generics_str = self.format_generics(generics,
                                                " {",
                                                self.block_indent + self.config.tab_spaces,
                                                codemap::mk_sp(span.lo,
                                                               body_start));
        self.changes.push_str_span(span, &generics_str);

        self.last_pos = body_start;
        self.block_indent += self.config.tab_spaces;
        for (i, f) in enum_def.variants.iter().enumerate() {
            let next_span_start: BytePos = if i == enum_def.variants.len() - 1 {
                span.hi
            } else {
                enum_def.variants[i + 1].span.lo
            };

            self.visit_variant(f, i == enum_def.variants.len() - 1, next_span_start);
        }
        self.block_indent -= self.config.tab_spaces;

        self.format_missing_with_indent(span.lo + BytePos(enum_snippet.rfind('}').unwrap() as u32));
        self.changes.push_str_span(span, "}");
    }

    // Variant of an enum
    fn visit_variant(&mut self, field: &ast::Variant, last_field: bool, next_span_start: BytePos) {
        if self.visit_attrs(&field.node.attrs) {
            return;
        }

        self.format_missing_with_indent(field.span.lo);

        let result = match field.node.kind {
            ast::VariantKind::TupleVariantKind(ref types) => {
                let vis = format_visibility(field.node.vis);
                self.changes.push_str_span(field.span, vis);
                let name = field.node.name.to_string();
                self.changes.push_str_span(field.span, &name);

                let mut result = String::new();

                if types.len() > 0 {
                    let items = itemize_list(self.codemap,
                                             Vec::new(),
                                             types.iter(),
                                             ",",
                                             ")",
                                             |arg| arg.ty.span.lo,
                                             |arg| arg.ty.span.hi,
                                             |arg| pprust::ty_to_string(&arg.ty),
                                             span_after(field.span, "(", self.codemap),
                                             next_span_start);

                    result.push('(');

                    let indent = self.block_indent
                                 + vis.len()
                                 + field.node.name.to_string().len()
                                 + 1; // Open paren

                    let comma_cost = if self.config.enum_trailing_comma { 1 } else { 0 };
                    let budget = self.config.ideal_width - indent - comma_cost - 1; // 1 = )

                    let fmt = ListFormatting {
                        tactic: ListTactic::HorizontalVertical,
                        separator: ",",
                        trailing_separator: SeparatorTactic::Never,
                        indent: indent,
                        h_width: budget,
                        v_width: budget,
                        ends_with_newline: false,
                    };
                    result.push_str(&write_list(&items, &fmt));
                    result.push(')');
                }

                if let Some(ref expr) = field.node.disr_expr {
                    result.push_str(" = ");
                    let expr_snippet = self.snippet(expr.span);
                    result.push_str(&expr_snippet);

                    // Make sure we do not exceed column limit
                    // 4 = " = ,"
                    assert!(
                        self.config.max_width >= vis.len() + name.len() + expr_snippet.len() + 4,
                        "Enum variant exceeded column limit");
                }

                result
            },
            ast::VariantKind::StructVariantKind(ref struct_def) => {
                // TODO Should limit the width, as we have a trailing comma
                self.format_struct("",
                                   field.node.name,
                                   field.node.vis,
                                   struct_def,
                                   None,
                                   field.span,
                                   self.block_indent)
            }
        };
        self.changes.push_str_span(field.span, &result);

        if !last_field || self.config.enum_trailing_comma {
            self.changes.push_str_span(field.span, ",");
        }

        self.last_pos = field.span.hi + BytePos(1);
    }

    fn format_struct(&self,
                     item_name: &str,
                     ident: ast::Ident,
                     vis: ast::Visibility,
                     struct_def: &ast::StructDef,
                     generics: Option<&ast::Generics>,
                     span: Span,
                     offset: usize)
                     -> String {
        let mut result = String::with_capacity(1024);

        let header_str = self.format_header(item_name, ident, vis);
        result.push_str(&header_str);

        if struct_def.fields.len() == 0 {
            result.push(';');
            return result;
        }

        let is_tuple = match struct_def.fields[0].node.kind {
            ast::StructFieldKind::NamedField(..) => false,
            ast::StructFieldKind::UnnamedField(..) => true
        };

        let (opener, terminator) = if is_tuple { ("(", ")") } else { (" {", "}") };

        let generics_str = match generics {
            Some(g) => self.format_generics(g,
                                            opener,
                                            offset + header_str.len(),
                                            codemap::mk_sp(span.lo,
                                                           struct_def.fields[0].span.lo)),
            None => opener.to_owned()
        };
        result.push_str(&generics_str);

        let items = itemize_list(self.codemap,
                                 Vec::new(),
                                 struct_def.fields.iter(),
                                 ",",
                                 terminator,
                                 |field| {
                                      // Include attributes and doc comments,
                                      // if present
                                      if field.node.attrs.len() > 0 {
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
        let used_budget = offset + header_str.len() + generics_str.len() + 3;

        // Conservative approximation
        let single_line_cost = (span.hi - struct_def.fields[0].span.lo).0;
        let break_line = !is_tuple || generics_str.contains('\n') ||
                         single_line_cost as usize + used_budget > self.config.max_width;

        if break_line {
            let indentation = make_indent(offset + self.config.tab_spaces);
            result.push('\n');
            result.push_str(&indentation);
        }

        let tactic = if break_line { ListTactic::Vertical } else { ListTactic::Horizontal };

        // 1 = ,
        let budget = self.config.ideal_width - offset + self.config.tab_spaces - 1;
        let fmt = ListFormatting { tactic: tactic,
                                   separator: ",",
                                   trailing_separator: self.config.struct_trailing_comma,
                                   indent: offset + self.config.tab_spaces,
                                   h_width: self.config.max_width,
                                   v_width: budget,
                                   ends_with_newline: false, };

        result.push_str(&write_list(&items, &fmt));

        if break_line {
            result.push('\n');
            result.push_str(&make_indent(offset));
        }

        result.push_str(terminator);

        if is_tuple {
            result.push(';');
        }

        result
    }

    pub fn visit_struct(&mut self,
                        ident: ast::Ident,
                        vis: ast::Visibility,
                        struct_def: &ast::StructDef,
                        generics: &ast::Generics,
                        span: Span) {
        let indent = self.block_indent;
        let result = self.format_struct("struct ",
                                        ident,
                                        vis,
                                        struct_def,
                                        Some(generics),
                                        span,
                                        indent);
        self.changes.push_str_span(span, &result);
        self.last_pos = span.hi;
    }

    fn format_header(&self, item_name: &str, ident: ast::Ident, vis: ast::Visibility) -> String {
        format!("{}{}{}", format_visibility(vis), item_name, &token::get_ident(ident))
    }

    fn format_generics(&self,
                       generics: &ast::Generics,
                       opener: &str,
                       offset: usize,
                       span: Span)
                       -> String {
        let mut result = self.rewrite_generics(generics, offset, span);

        if generics.where_clause.predicates.len() > 0 || result.contains('\n') {
            result.push_str(&self.rewrite_where_clause(&generics.where_clause,
                                                       self.block_indent,
                                                       span.hi));
            result.push_str(&make_indent(self.block_indent));
            result.push('\n');
            result.push_str(opener.trim());
        } else {
            result.push_str(opener);
        }

        result
    }

    // Field of a struct
    fn format_field(&self, field: &ast::StructField) -> String {
        if contains_skip(&field.node.attrs) {
            return self.snippet(codemap::mk_sp(field.node.attrs[0].span.lo, field.span.hi));
        }

        let name = match field.node.kind {
            ast::StructFieldKind::NamedField(ident, _) => Some(token::get_ident(ident)),
            ast::StructFieldKind::UnnamedField(_) => None,
        };
        let vis = match field.node.kind {
            ast::StructFieldKind::NamedField(_, vis) |
            ast::StructFieldKind::UnnamedField(vis) => format_visibility(vis)
        };
        let typ = pprust::ty_to_string(&field.node.ty);

        let indent = self.block_indent + self.config.tab_spaces;
        let mut attr_str = self.rewrite_attrs(&field.node.attrs, indent);
        if attr_str.len() > 0 {
            attr_str.push('\n');
            attr_str.push_str(&make_indent(indent));
        }

        match name {
            Some(name) => format!("{}{}{}: {}", attr_str, vis, name, typ),
            None => format!("{}{}{}", attr_str, vis, typ)
        }
    }

    fn rewrite_generics(&self, generics: &ast::Generics, offset: usize, span: Span) -> String {
        // FIXME convert bounds to where clauses where they get too big or if
        // there is a where clause at all.
        let mut result = String::new();
        let lifetimes: &[_] = &generics.lifetimes;
        let tys: &[_] = &generics.ty_params;
        if lifetimes.len() + tys.len() == 0 {
            return result;
        }

        let budget = self.config.max_width - offset - 2;
        // TODO might need to insert a newline if the generics are really long
        result.push('<');

        // Strings for the generics.
        let lt_strs = lifetimes.iter().map(|l| self.rewrite_lifetime_def(l));
        let ty_strs = tys.iter().map(|ty| self.rewrite_ty_param(ty));

        // Extract comments between generics.
        let lt_spans = lifetimes.iter().map(|l| {
            let hi = if l.bounds.len() == 0 {
                l.lifetime.span.hi
            } else {
                l.bounds[l.bounds.len() - 1].span.hi
            };
            codemap::mk_sp(l.lifetime.span.lo, hi)
        });
        let ty_spans = tys.iter().map(span_for_ty_param);

        let mut items = itemize_list(self.codemap,
                                     Vec::new(),
                                     lt_spans.chain(ty_spans),
                                     ",",
                                     ">",
                                     |sp| sp.lo,
                                     |sp| sp.hi,
                                     |_| String::new(),
                                     span_after(span, "<", self.codemap),
                                     span.hi);

        for (item, ty) in items.iter_mut().zip(lt_strs.chain(ty_strs)) {
            item.item = ty;
        }

        let fmt = ListFormatting { tactic: ListTactic::HorizontalVertical,
                                   separator: ",",
                                   trailing_separator: SeparatorTactic::Never,
                                   indent: offset + 1,
                                   h_width: budget,
                                   v_width: budget,
                                   ends_with_newline: true, };
        result.push_str(&write_list(&items, &fmt));

        result.push('>');

        result
    }

    fn rewrite_where_clause(&self,
                            where_clause: &ast::WhereClause,
                            indent: usize,
                            span_end: BytePos)
                            -> String {
        let mut result = String::new();
        if where_clause.predicates.len() == 0 {
            return result;
        }

        result.push('\n');
        result.push_str(&make_indent(indent + 4));
        result.push_str("where ");

        let span_start = span_for_where_pred(&where_clause.predicates[0]).lo;
        let items = itemize_list(self.codemap,
                                 Vec::new(),
                                 where_clause.predicates.iter(),
                                 ",",
                                 "{",
                                 |pred| span_for_where_pred(pred).lo,
                                 |pred| span_for_where_pred(pred).hi,
                                 |pred| self.rewrite_pred(pred),
                                 span_start,
                                 span_end);

        let budget = self.config.ideal_width + self.config.leeway - indent - 10;
        let fmt = ListFormatting { tactic: ListTactic::Vertical,
                                   separator: ",",
                                   trailing_separator: SeparatorTactic::Never,
                                   indent: indent + 10,
                                   h_width: budget,
                                   v_width: budget,
                                   ends_with_newline: true, };
        result.push_str(&write_list(&items, &fmt));

        result
    }

    fn rewrite_return(&self, ret: &ast::FunctionRetTy) -> String {
        match *ret {
            ast::FunctionRetTy::DefaultReturn(_) => String::new(),
            ast::FunctionRetTy::NoReturn(_) => "-> !".to_owned(),
            ast::FunctionRetTy::Return(ref ty) => "-> ".to_owned() + &pprust::ty_to_string(ty),
        }
    }

    // TODO we farm this out, but this could spill over the column limit, so we
    // ought to handle it properly.
    fn rewrite_fn_input(&self, arg: &ast::Arg) -> String {
        format!("{}: {}",
                pprust::pat_to_string(&arg.pat),
                pprust::ty_to_string(&arg.ty))
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
    if ty.bounds.len() == 0 {
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
