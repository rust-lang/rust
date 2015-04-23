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
                      abi: &abi::Abi,
                      vis: ast::Visibility,
                      next_span: Span) // next_span is a nasty hack, its a loose upper
                                       // bound on any comments after the where clause.
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
        let generics_indent = indent + result.len();
        result.push_str(&self.rewrite_generics(generics,
                                               generics_indent,
                                               span_for_return(&fd.output)));

        let ret_str = self.rewrite_return(&fd.output);

        // Args.
        let (one_line_budget, multi_line_budget, arg_indent) =
            self.compute_budgets_for_args(&mut result, indent, ret_str.len(), newline_brace);

        result.push('(');
        result.push_str(&self.rewrite_args(&fd.inputs,
                                           explicit_self,
                                           one_line_budget,
                                           multi_line_budget,
                                           arg_indent,
                                           span_for_return(&fd.output)));
        result.push(')');

        // Where clause.
        result.push_str(&self.rewrite_where_clause(where_clause, indent, next_span));

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

        // TODO any comments here?

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

    fn rewrite_args(&self,
                    args: &[ast::Arg],
                    explicit_self: Option<&ast::ExplicitSelf>,
                    one_line_budget: usize,
                    multi_line_budget: usize,
                    arg_indent: usize,
                    ret_span: Span)
        -> String
    {
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
                        &ast::Mutability::MutMutable => "mut ".to_string(),
                        &ast::Mutability::MutImmutable => String::new(),
                    };
                    arg_item_strs[0] = format!("&{}{}self", lt_str, mut_str);
                    min_args = 2;
                }
                ast::ExplicitSelf_::SelfExplicit(ref ty, _) => {
                    arg_item_strs[0] = format!("self: {}", pprust::ty_to_string(ty));
                }
                ast::ExplicitSelf_::SelfValue(_) => {
                    arg_item_strs[0] = "self".to_string();
                    min_args = 2;
                }
                _ => {}
            }
        }

        // Comments between args
        let mut arg_comments = Vec::new();
        if min_args == 2 {
            arg_comments.push("".to_string());
        }
        // TODO if there are no args, there might still be a comment, but without
        // spans for the comment or parens, there is no chance of getting it right.
        // You also don't get to put a comment on self, unless it is explicit.
        if args.len() >= min_args {
            arg_comments = self.make_comments_for_list(arg_comments,
                                                       args[min_args-1..].iter(),
                                                       ",",
                                                       ")",
                                                       |arg| arg.pat.span.lo,
                                                       |arg| arg.ty.span.hi,
                                                       ret_span.lo);
        }

        debug!("comments: {:?}", arg_comments);

        // If there are // comments, keep them multi-line.
        let mut list_tactic = ListTactic::HorizontalVertical;
        if arg_comments.iter().any(|c| c.contains("//")) {
            list_tactic = ListTactic::Vertical;
        }

        assert_eq!(arg_item_strs.len(), arg_comments.len());
        let arg_strs: Vec<_> = arg_item_strs.into_iter().zip(arg_comments.into_iter()).collect();

        let fmt = ListFormatting {
            tactic: list_tactic,
            separator: ",",
            trailing_separator: SeparatorTactic::Never,
            indent: arg_indent,
            h_width: one_line_budget,
            v_width: multi_line_budget,
        };

        write_list(&arg_strs, &fmt)
    }

    // Gets comments in between items of a list. 
    fn make_comments_for_list<T, I, F1, F2>(&self,
                                            prefix: Vec<String>,
                                            mut it: I,
                                            separator: &str,
                                            terminator: &str,
                                            get_lo: F1,
                                            get_hi: F2,
                                            next_span_start: BytePos)
        -> Vec<String>
        where I: Iterator<Item=T>,
              F1: Fn(&T) -> BytePos,
              F2: Fn(&T) -> BytePos
    {
        let mut result = prefix;

        let mut prev_end = get_hi(&it.next().unwrap());
        for item in it {
            let cur_start = get_lo(&item);
            let snippet = self.snippet(codemap::mk_sp(prev_end, cur_start));
            let mut snippet = snippet.trim();
            if snippet.starts_with(separator) {
                snippet = snippet[1..].trim();
            } else if snippet.ends_with(separator) {
                snippet = snippet[..snippet.len()-1].trim();
            }
            result.push(snippet.to_string());
            prev_end = get_hi(&item);
        }
        // Get the last commment.
        // FIXME If you thought the crap with the commas was ugly, just wait.
        // This is awful. We're going to look from the last item span to the
        // start of the return type span, then we drop everything after the
        // first closing paren. Obviously, this will break if there is a 
        // closing paren in the comment.
        // The fix is comments in the AST or a span for the closing paren.
        let snippet = self.snippet(codemap::mk_sp(prev_end, next_span_start));
        let snippet = snippet.trim();
        let snippet = &snippet[..snippet.find(terminator).unwrap()];
        let snippet = snippet.trim();
        result.push(snippet.to_string());

        result
    }

    fn compute_budgets_for_args(&self,
                                result: &mut String,
                                indent: usize,
                                ret_str_len: usize,
                                newline_brace: bool)
        -> (usize, usize, usize)
    {
        let mut budgets = None;

        // Try keeping everything on the same line
        if !result.contains("\n") {
            // 3 = `() `, space is before ret_string
            let mut used_space = indent + result.len() + 3 + ret_str_len;
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

        budgets.unwrap()
    }

    fn newline_for_brace(&self, where_clause: &ast::WhereClause) -> bool {
        match FN_BRACE_STYLE {
            BraceStyle::AlwaysNextLine => true,
            BraceStyle::SameLineWhere if where_clause.predicates.len() > 0 => true,
            _ => false,
        }
    }

    fn rewrite_generics(&self, generics: &ast::Generics, indent: usize, ret_span: Span) -> String {
        // FIXME convert bounds to where clauses where they get too big or if
        // there is a where clause at all.
        let mut result = String::new();
        let lifetimes: &[_] = &generics.lifetimes;
        let tys: &[_] = &generics.ty_params;
        if lifetimes.len() + tys.len() == 0 {
            return result;
        }

        let budget = MAX_WIDTH - indent - 2;
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
        let comments = self.make_comments_for_list(Vec::new(),
                                                   lt_spans.chain(ty_spans),
                                                   ",",
                                                   ">",
                                                   |sp| sp.lo,
                                                   |sp| sp.hi,
                                                   ret_span.lo);

        // If there are // comments, keep them multi-line.
        let mut list_tactic = ListTactic::HorizontalVertical;
        if comments.iter().any(|c| c.contains("//")) {
            list_tactic = ListTactic::Vertical;
        }

        let generics_strs: Vec<_> = lt_strs.chain(ty_strs).zip(comments.into_iter()).collect();
        let fmt = ListFormatting {
            tactic: list_tactic,
            separator: ",",
            trailing_separator: SeparatorTactic::Never,
            indent: indent + 1,
            h_width: budget,
            v_width: budget,
        };
        result.push_str(&write_list(&generics_strs, &fmt));

        result.push('>');

        result
    }

    fn rewrite_where_clause(&self,
                            where_clause: &ast::WhereClause,
                            indent: usize,
                            next_span: Span)
        -> String
    {
        let mut result = String::new();
        if where_clause.predicates.len() == 0 {
            return result;
        }

        result.push('\n');
        result.push_str(&make_indent(indent + 4));
        result.push_str("where ");

        let comments = vec![String::new(); where_clause.predicates.len()];
        // TODO uncomment when spans are fixed
        // println!("{:?} {:?}", where_clause.predicates.iter().map(|p| self.snippet(span_for_where_pred(p))).collect::<Vec<_>>(), next_span.lo);
        // let comments = self.make_comments_for_list(Vec::new(),
        //                                            where_clause.predicates.iter(),
        //                                            ",",
        //                                            "{",
        //                                            |pred| span_for_where_pred(pred).lo,
        //                                            |pred| span_for_where_pred(pred).hi,
        //                                            next_span.lo);
        let where_strs: Vec<_> = where_clause.predicates.iter()
                                                        .map(|p| (self.rewrite_pred(p)))
                                                        .zip(comments.into_iter())
                                                        .collect();

        let budget = IDEAL_WIDTH + LEEWAY - indent - 10;
        let fmt = ListFormatting {
            tactic: ListTactic::Vertical,
            separator: ",",
            trailing_separator: SeparatorTactic::Never,
            indent: indent + 10,
            h_width: budget,
            v_width: budget,
        };
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
