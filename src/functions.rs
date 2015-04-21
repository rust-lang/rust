// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use {ReturnIndent, MAX_WIDTH, BraceStyle,
     IDEAL_WIDTH, LEEWAY, FN_BRACE_STYLE, FN_RETURN_INDENT};
use utils::make_indent;
use lists::{write_list, ListFormatting, SeparatorTactic, ListTactic};
use visitor::FmtVisitor;
use syntax::{ast, abi};
use syntax::print::pprust;
use syntax::parse::token;

impl<'a> FmtVisitor<'a> {
    // TODO extract methods for args and generics
    pub fn rewrite_fn(&mut self,
                      indent: usize,
                      ident: ast::Ident,
                      fd: &ast::FnDecl,
                      explicit_self: Option<&ast::ExplicitSelf>,
                      generics: &ast::Generics,
                      unsafety: &ast::Unsafety,
                      abi: &abi::Abi,
                      vis: ast::Visibility)
        -> String
    {
        // FIXME we'll lose any comments in between parts of the function decl, but anyone
        // who comments there probably deserves what they get.

        let where_clause = &generics.where_clause;
        let newline_brace = self.newline_for_brace(where_clause);

        let mut result = String::with_capacity(1024);
        // Vis unsafety abi.
        if vis == ast::Visibility::Public {
            result.push_str("pub ");
        }
        if let &ast::Unsafety::Unsafe = unsafety {
            result.push_str("unsafe ");
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
        // FIXME convert bounds to where clauses where they get too big or if
        // there is a where clause at all.
        let lifetimes: &[_] = &generics.lifetimes;
        let tys: &[_] = &generics.ty_params;
        if lifetimes.len() + tys.len() > 0 {
            let budget = MAX_WIDTH - indent - result.len() - 2;
            // TODO might need to insert a newline if the generics are really long
            result.push('<');

            let lt_strs = lifetimes.iter().map(|l| self.rewrite_lifetime_def(l));
            let ty_strs = tys.iter().map(|ty| self.rewrite_ty_param(ty));
            let generics_strs: Vec<_> = lt_strs.chain(ty_strs).map(|s| (s, String::new())).collect();
            let fmt = ListFormatting {
                tactic: ListTactic::HorizontalVertical,
                separator: ",",
                trailing_separator: SeparatorTactic::Never,
                indent: indent + result.len() + 1,
                h_width: budget,
                v_width: budget,
            };
            result.push_str(&write_list(&generics_strs, &fmt));

            result.push('>');
        }

        let ret_str = self.rewrite_return(&fd.output);

        // Args.
        let args = &fd.inputs;

        let mut budgets = None;

        // Try keeping everything on the same line
        if !result.contains("\n") {
            // 3 = `() `, space is before ret_string
            let mut used_space = indent + result.len() + 3 + ret_str.len();
            if newline_brace {
                used_space += 2;
            }
            let one_line_budget = if used_space > MAX_WIDTH {
                0
            } else {
                MAX_WIDTH - used_space
            };

            let used_space = indent + result.len() + 2;
            let max_space = IDEAL_WIDTH + LEEWAY;
            if used_space < max_space {
                budgets = Some((one_line_budget,
                                // 2 = `()`
                                max_space - used_space,
                                indent + result.len() + 1));
            }
        }

        // Didn't work. we must force vertical layout and put args on a newline.
        if let None = budgets {
            result.push('\n');
            result.push_str(&make_indent(indent + 4));
            // 6 = new indent + `()`
            let used_space = indent + 6;
            let max_space = IDEAL_WIDTH + LEEWAY;
            if used_space > max_space {
                // Whoops! bankrupt.
                // TODO take evasive action, perhaps kill the indent or something.
            } else {
                // 5 = new indent + `(`
                budgets = Some((0, max_space - used_space, indent + 5));
            }
        }

        let (one_line_budget, multi_line_budget, arg_indent) = budgets.unwrap();
        result.push('(');

        let fmt = ListFormatting {
            tactic: ListTactic::HorizontalVertical,
            separator: ",",
            trailing_separator: SeparatorTactic::Never,
            indent: arg_indent,
            h_width: one_line_budget,
            v_width: multi_line_budget,
        };
        // TODO dead spans
        let mut arg_strs: Vec<_> = args.iter().map(|a| (self.rewrite_fn_input(a), String::new())).collect();
        // Account for sugary self.
        if let Some(explicit_self) = explicit_self {
            match explicit_self.node {
                ast::ExplicitSelf_::SelfRegion(ref lt, ref m, _) => {
                    let lt_str = match lt {
                        &Some(ref l) => format!("{} ", pprust::lifetime_to_string(l)),
                        &None => String::new(),
                    };
                    let mut_str = match m {
                        &ast::Mutability::MutMutable => "mut ".to_string(),
                        &ast::Mutability::MutImmutable => String::new(),
                    };
                    arg_strs[0].0 = format!("&{}{}self", lt_str, mut_str);
                }
                ast::ExplicitSelf_::SelfExplicit(ref ty, _) => {
                    arg_strs[0].0 = format!("self: {}", pprust::ty_to_string(ty));
                }
                ast::ExplicitSelf_::SelfValue(_) => {
                    arg_strs[0].0 = "self".to_string();
                }
                _ => {}
            }
        }
        result.push_str(&write_list(&arg_strs, &fmt));

        result.push(')');

        // Where clause.
        result.push_str(&self.rewrite_where_clause(where_clause, indent));

        // Return type.
        if ret_str.len() > 0 {
            // If we've already gone multi-line, or the return type would push
            // over the max width, then put the return type on a new line.
            if result.contains("\n") ||
               result.len() + indent + ret_str.len() > MAX_WIDTH {
                let indent = match FN_RETURN_INDENT {
                    ReturnIndent::WithWhereClause => indent + 4,
                    // TODO we might want to check that using the arg indent doesn't
                    // blow our budget, and if it does, then fallback to the where
                    // clause indent.
                    ReturnIndent::WithArgs => arg_indent,
                };

                result.push('\n');
                result.push_str(&make_indent(indent));
            } else {
                result.push(' ');
            }
            result.push_str(&ret_str);
        }

        // Prepare for the function body by possibly adding a newline and indent.
        // FIXME we'll miss anything between the end of the signature and the start
        // of the body, but we need more spans from the compiler to solve this.
        if newline_brace {
            result.push('\n');
            result.push_str(&make_indent(self.block_indent));
        } else {
            result.push(' ');
        }

        result
    }

    fn newline_for_brace(&self, where_clause: &ast::WhereClause) -> bool {
        match FN_BRACE_STYLE {
            BraceStyle::AlwaysNextLine => true,
            BraceStyle::SameLineWhere if where_clause.predicates.len() > 0 => true,
            _ => false,
        }
    }

    fn rewrite_where_clause(&self, where_clause: &ast::WhereClause, indent: usize) -> String {
        let mut result = String::new();
        if where_clause.predicates.len() == 0 {
            return result;
        }

        result.push('\n');
        result.push_str(&make_indent(indent + 4));
        result.push_str("where ");

        let budget = IDEAL_WIDTH + LEEWAY - indent - 10;
        let fmt = ListFormatting {
            tactic: ListTactic::Vertical,
            separator: ",",
            trailing_separator: SeparatorTactic::Never,
            indent: indent + 10,
            h_width: budget,
            v_width: budget,
        };
        let where_strs: Vec<_> = where_clause.predicates.iter().map(|p| (self.rewrite_pred(p), String::new())).collect();
        result.push_str(&write_list(&where_strs, &fmt));

        result
    }

    fn rewrite_return(&self, ret: &ast::FunctionRetTy) -> String {
        match *ret {
            ast::FunctionRetTy::DefaultReturn(_) => String::new(),
            ast::FunctionRetTy::NoReturn(_) => "-> !".to_string(),
            ast::FunctionRetTy::Return(ref ty) => "-> ".to_string() + &pprust::ty_to_string(ty),
        }        
    }

    // TODO we farm this out, but this could spill over the column limit, so we ought to handle it properly
    fn rewrite_fn_input(&self, arg: &ast::Arg) -> String {
        format!("{}: {}",
                pprust::pat_to_string(&arg.pat),
                pprust::ty_to_string(&arg.ty))
    }
}
