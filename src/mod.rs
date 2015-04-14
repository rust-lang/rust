// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]
#![feature(box_patterns)]
#![feature(rustc_private)]
#![feature(collections)]
#![feature(exit_status)]
#![feature(str_char)]

// TODO we're going to allocate a whole bunch of temp Strings, is it worth
// keeping some scratch mem for this and running our own StrPool?
// TODO for lint violations of names, emit a refactor script

// TODO priorities
// Fix fns and methods properly - need visibility in visit
// Writing output
// Working on multiple files, inclding empty ones
// Smoke testing till we can use it

#[macro_use]
extern crate log;

extern crate getopts;
extern crate rustc;
extern crate rustc_driver;
extern crate syntax;

extern crate strings;

use rustc::session::Session;
use rustc::session::config::{self, Input};
use rustc_driver::{driver, CompilerCalls, Compilation};

use syntax::{ast, ptr, abi};
use syntax::codemap::{self, CodeMap, Span, Pos, BytePos};
use syntax::diagnostics;
use syntax::parse::token;
use syntax::print::pprust;
use syntax::visit;

use std::path::PathBuf;

use changes::ChangeSet;

mod changes;

const IDEAL_WIDTH: usize = 80;
const LEEWAY: usize = 5;
const MAX_WIDTH: usize = 100;
const MIN_STRING: usize = 10;
const TAB_SPACES: usize = 4;

// Formatting which depends on the AST.
fn fmt_ast<'a>(krate: &ast::Crate, codemap: &'a CodeMap) -> ChangeSet<'a> {
    let mut visitor = FmtVisitor::from_codemap(codemap);
    visit::walk_crate(&mut visitor, krate);
    let files = codemap.files.borrow();
    if let Some(last) = files.last() {
        visitor.format_missing(last.end_pos);
    }

    visitor.changes
}

// Formatting done on a char by char basis.
fn fmt_lines(changes: &mut ChangeSet) {
    // Iterate over the chars in the change set.
    for (f, text) in changes.text() {
        let mut trims = vec![];
        let mut last_wspace: Option<usize> = None;
        let mut line_len = 0;
        let mut cur_line = 1;
        for (c, b) in text.chars() {
            if c == '\n' { // TOOD test for \r too
                // Check for (and record) trailing whitespace.
                if let Some(lw) = last_wspace {
                    trims.push((cur_line, lw, b));
                    line_len -= b - lw;
                }
                // Check for any line width errors we couldn't correct.
                if line_len > MAX_WIDTH {
                    // FIXME store the error rather than reporting immediately.
                    println!("Rustfmt couldn't fix (sorry). {}:{}: line longer than {} characters",
                             f, cur_line, MAX_WIDTH);
                }
                line_len = 0;
                cur_line += 1;
                last_wspace = None;
            } else {
                line_len += 1;
                if c.is_whitespace() {
                    if last_wspace.is_none() {
                        last_wspace = Some(b);
                    }
                } else {
                    last_wspace = None;
                }
            }
        }

        for &(l, _, _) in trims.iter() {
            // FIXME store the error rather than reporting immediately.
            println!("Rustfmt left trailing whitespace at {}:{} (sorry)", f, l);
        }
    }
}

struct FmtVisitor<'a> {
    codemap: &'a CodeMap,
    changes: ChangeSet<'a>,
    last_pos: BytePos,
    // TODO RAII util for indenting
    block_indent: usize,
}

impl<'a, 'v> visit::Visitor<'v> for FmtVisitor<'a> {
    fn visit_expr(&mut self, ex: &'v ast::Expr) {
        // TODO uncomment
        // debug!("visit_expr: {:?} {:?}",
        //        self.codemap.lookup_char_pos(ex.span.lo),
        //        self.codemap.lookup_char_pos(ex.span.hi));
        self.format_missing(ex.span.lo);
        let offset = self.changes.cur_offset_span(ex.span);
        let new_str = self.rewrite_expr(ex, MAX_WIDTH - offset, offset);
        self.changes.push_str_span(ex.span, &new_str);
        self.last_pos = ex.span.hi;
    }

    fn visit_block(&mut self, b: &'v ast::Block) {
        // TODO uncomment
        // debug!("visit_block: {:?} {:?}",
        //        self.codemap.lookup_char_pos(b.span.lo),
        //        self.codemap.lookup_char_pos(b.span.hi));
        self.format_missing(b.span.lo);

        self.changes.push_str_span(b.span, "{");
        self.last_pos = self.last_pos + BytePos(1);
        self.block_indent += TAB_SPACES;

        for stmt in &b.stmts {
            self.format_missing_with_indent(stmt.span.lo);
            self.visit_stmt(&stmt)
        }
        match b.expr {
            Some(ref e) => {
                self.format_missing_with_indent(e.span.lo);
                self.visit_expr(e);
            }
            None => {}
        }

        self.block_indent -= TAB_SPACES;
        // TODO we should compress any newlines here to just one
        self.format_missing_with_indent(b.span.hi - BytePos(1));
        self.changes.push_str_span(b.span, "}");
        self.last_pos = b.span.hi;
    }

    fn visit_fn(&mut self,
                fk: visit::FnKind<'v>,
                fd: &'v ast::FnDecl,
                b: &'v ast::Block,
                s: Span,
                _: ast::NodeId) {
        // TODO need to get the visibility from somewhere
        self.format_missing(s.lo);
        self.last_pos = s.lo;

        // TODO need to check against expected indent
        let indent = self.codemap.lookup_char_pos(s.lo).col.0;
        match fk {
            visit::FkItemFn(ident, ref generics, ref unsafety, ref abi) => {
                let new_fn = self.rewrite_fn(indent,
                                             ident,
                                             fd,
                                             None,
                                             generics,
                                             unsafety,
                                             abi,
                                             ast::Visibility::Inherited);
                self.changes.push_str_span(s, &new_fn);
            }
            visit::FkMethod(ident, ref sig) => {
                let new_fn = self.rewrite_fn(indent,
                                             ident,
                                             fd,
                                             Some(&sig.explicit_self),
                                             &sig.generics,
                                             &sig.unsafety,
                                             &sig.abi,
                                             ast::Visibility::Inherited);
                self.changes.push_str_span(s, &new_fn);
            }
            visit::FkFnBlock(..) => {}
        }

        // FIXME we'll miss anything between the end of the signature and the start
        // of the body, but we need more spans from the compiler to solve this.
        self.changes.push_str_span(s, "\n");
        self.changes.push_str_span(s, &make_indent(self.block_indent));
        self.last_pos = b.span.lo;
        self.visit_block(b)
    }

    fn visit_item(&mut self, item: &'v ast::Item) {
        match item.node {
            ast::Item_::ItemUse(ref vp) => {
                match vp.node {
                    ast::ViewPath_::ViewPathList(ref path, ref path_list) => {
                        self.format_missing(item.span.lo);
                        let new_str = self.rewrite_use_list(path, path_list, vp.span);
                        self.changes.push_str_span(item.span, &new_str);
                        self.last_pos = item.span.hi;
                    }
                    ast::ViewPath_::ViewPathGlob(_) => {
                        // FIXME convert to list?
                    }
                    _ => {}
                }
                visit::walk_item(self, item);
            }
            ast::Item_::ItemImpl(..) => {
                self.block_indent += TAB_SPACES;
                visit::walk_item(self, item);
                self.block_indent -= TAB_SPACES;
            }
            _ => {
                visit::walk_item(self, item);
            }
        }
    }

    fn visit_mac(&mut self, mac: &'v ast::Mac) {
        visit::walk_mac(self, mac)
    }
}

fn make_indent(width: usize) -> String {
    let mut indent = String::with_capacity(width);
    for _ in 0..width {
        indent.push(' ')
    }
    indent
}

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
enum ListTactic {
    // One item per row.
    Vertical,
    // All items on one row.
    Horizontal,
    // Try Horizontal layout, if that fails then vertical
    HorizontalVertical,
    // Pack as many items as possible per row over (possibly) many rows.
    Mixed,
}

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
enum SeparatorTactic {
    Always,
    Never,
    Vertical,
}

struct ListFormatting<'a> {
    tactic: ListTactic,
    separator: &'a str,
    trailing_separator: SeparatorTactic,
    indent: usize,
    // Available width if we layout horizontally.
    h_width: usize,
    // Available width if we layout vertically
    v_width: usize,
}

// Format a list of strings into a string.
fn write_list<'b>(items:&[(String, String)], formatting: &ListFormatting<'b>) -> String {
    if items.len() == 0 {
        return String::new();
    }

    let mut tactic = formatting.tactic;

    let h_width = formatting.h_width;
    let v_width = formatting.v_width;
    let sep_len = formatting.separator.len();

    // Conservatively overestimates because of the changing separator tactic.
    let sep_count = if formatting.trailing_separator != SeparatorTactic::Never {
        items.len()
    } else {
        items.len() - 1
    };

    // TODO count dead space too.
    let total_width = items.iter().map(|&(ref s, _)| s.len()).fold(0, |a, l| a + l);

    // Check if we need to fallback from horizontal listing, if possible.
    if tactic == ListTactic::HorizontalVertical { 
        if (total_width + (sep_len + 1) * sep_count) > h_width {
            tactic = ListTactic::Vertical;
        } else {
            tactic = ListTactic::Horizontal;
        }
    }

    // Now that we know how we will layout, we can decide for sure if there
    // will be a trailing separator.
    let trailing_separator = match formatting.trailing_separator {
        SeparatorTactic::Always => true,
        SeparatorTactic::Vertical => tactic == ListTactic::Vertical,
        SeparatorTactic::Never => false,
    };

    // Create a buffer for the result.
    // TODO could use a StringBuffer or rope for this
    let alloc_width = if tactic == ListTactic::Horizontal {
        total_width + (sep_len + 1) * sep_count
    } else {
        total_width + items.len() * (formatting.indent + 1)
    };
    let mut result = String::with_capacity(alloc_width);

    let mut line_len = 0;
    let indent_str = &make_indent(formatting.indent);
    for (i, &(ref item, _)) in items.iter().enumerate() {
        let first = i == 0;
        let separate = i != items.len() - 1 || trailing_separator;

        match tactic {
            ListTactic::Horizontal if !first => {
                result.push(' ');
            }
            ListTactic::Vertical if !first => {
                result.push('\n');
                result.push_str(indent_str);
            }
            ListTactic::Mixed => {
                let mut item_width = item.len();
                if separate {
                    item_width += sep_len;
                }

                if line_len > 0 && line_len + item_width > v_width {
                    result.push('\n');
                    result.push_str(indent_str);
                    line_len = 0;
                }

                if line_len > 0 {
                    result.push(' ');
                    line_len += 1;
                }

                line_len += item_width;
            }
            _ => {}
        }

        result.push_str(item);
        
        if separate {
            result.push_str(formatting.separator);
        }
        // TODO dead spans
    }

    result
}

impl<'a> FmtVisitor<'a> {
    fn from_codemap<'b>(codemap: &'b CodeMap) -> FmtVisitor<'b> {
        FmtVisitor {
            codemap: codemap,
            changes: ChangeSet::from_codemap(codemap),
            last_pos: BytePos(0),
            block_indent: 0,
        }
    }

    // TODO these format_missing methods are ugly. Refactor and add unit tests
    // for the central whitespace stripping loop.
    fn format_missing(&mut self, end: BytePos) {
        self.format_missing_inner(end, |this, last_snippet, span, _| {
            this.changes.push_str_span(span, last_snippet)
        })
    }

    fn format_missing_with_indent(&mut self, end: BytePos) {
        self.format_missing_inner(end, |this, last_snippet, span, snippet| {
            if last_snippet == snippet {
                // No new lines
                this.changes.push_str_span(span, last_snippet);
                this.changes.push_str_span(span, "\n");
            } else {
                this.changes.push_str_span(span, last_snippet.trim_right());
            }
            let indent = make_indent(this.block_indent);
            this.changes.push_str_span(span, &indent);
        })
    }

    fn format_missing_inner<F: Fn(&mut FmtVisitor, &str, Span, &str)>(&mut self,
                                                                      end: BytePos,
                                                                      process_last_snippet: F)
    {
        let start = self.last_pos;
        // TODO uncomment
        // debug!("format_missing_inner: {:?} to {:?}",
        //        self.codemap.lookup_char_pos(start),
        //        self.codemap.lookup_char_pos(end));

        // TODO(#11) gets tricky if we're missing more than one file
        // assert!(self.codemap.lookup_char_pos(start).file.name == self.codemap.lookup_char_pos(end).file.name,
        //         "not implemented: unformated span across files: {} and {}",
        //         self.codemap.lookup_char_pos(start).file.name,
        //         self.codemap.lookup_char_pos(end).file.name);
        // assert!(start <= end,
        //         "Request to format inverted span: {:?} to {:?}",
        //         self.codemap.lookup_char_pos(start),
        //         self.codemap.lookup_char_pos(end));

        if start == end {
            return;
        }

        self.last_pos = end;
        let span = codemap::mk_sp(start, end);
        let snippet = self.snippet(span);

        // Trim whitespace from the right hand side of each line.
        // Annoyingly, the library functions for splitting by lines etc. are not
        // quite right, so we must do it ourselves.
        let mut line_start = 0;
        let mut last_wspace = None;
        for (i, c) in snippet.char_indices() {
            if c == '\n' {
                if let Some(lw) = last_wspace {
                    self.changes.push_str_span(span, &snippet[line_start..lw]);
                    self.changes.push_str_span(span, "\n");
                } else {
                    self.changes.push_str_span(span, &snippet[line_start..i+1]);
                }

                line_start = i + 1;
                last_wspace = None;
            } else {
                if c.is_whitespace() {
                    if last_wspace.is_none() {
                        last_wspace = Some(i);
                    }
                } else {
                    last_wspace = None;
                }
            }
        }
        process_last_snippet(self, &snippet[line_start..], span, &snippet);
    }

    fn snippet(&self, span: Span) -> String {
        match self.codemap.span_to_snippet(span) {
            Ok(s) => s,
            Err(_) => {
                println!("Couldn't make snippet for span {:?}", span);
                "".to_string()
            }
        }
    }

    // TODO NEEDS TESTS
    fn rewrite_string_lit(&mut self, s: &str, span: Span, width: usize, offset: usize) -> String {
        // FIXME I bet this stomps unicode escapes in the source string

        // Check if there is anything to fix: we always try to fixup multi-line
        // strings, or if the string is too long for the line.
        let l_loc = self.codemap.lookup_char_pos(span.lo);
        let r_loc = self.codemap.lookup_char_pos(span.hi);
        if l_loc.line == r_loc.line && r_loc.col.to_usize() <= MAX_WIDTH {
            return self.snippet(span);
        }

        // TODO if lo.col > IDEAL - 10, start a new line (need cur indent for that)

        let s = s.escape_default();

        let offset = offset + 1;
        let indent = make_indent(offset);
        let indent = &indent;

        let max_chars = width - 1;

        let mut cur_start = 0;
        let mut result = String::new();
        result.push('"');
        loop {
            let mut cur_end = cur_start + max_chars;

            if cur_end >= s.len() {
                result.push_str(&s[cur_start..]);
                break;
            }

            // Make sure we're on a char boundary.
            cur_end = next_char(&s, cur_end);

            // Push cur_end left until we reach whitespace
            while !s.char_at(cur_end-1).is_whitespace() {
                cur_end = prev_char(&s, cur_end);

                if cur_end - cur_start < MIN_STRING {
                    // We can't break at whitespace, fall back to splitting
                    // anywhere that doesn't break an escape sequence
                    cur_end = next_char(&s, cur_start + max_chars);
                    while s.char_at(cur_end) == '\\' {
                        cur_end = prev_char(&s, cur_end);
                    }
                }
            }
            // Make sure there is no whitespace to the right of the break.
            while cur_end < s.len() && s.char_at(cur_end).is_whitespace() {
                cur_end = next_char(&s, cur_end+1);
            }
            result.push_str(&s[cur_start..cur_end]);
            result.push_str("\\\n");
            result.push_str(indent);

            cur_start = cur_end;
        }
        result.push('"');

        result
    }

    // Basically just pretty prints a multi-item import.
    fn rewrite_use_list(&mut self,
                        path: &ast::Path,
                        path_list: &[ast::PathListItem],
                        vp_span: Span) -> String {
        // FIXME remove unused imports

        // FIXME check indentation
        let l_loc = self.codemap.lookup_char_pos(vp_span.lo);

        let path_str = pprust::path_to_string(&path);

        // 3 = :: + {
        let indent = l_loc.col.0 + path_str.len() + 3;
        let fmt = ListFormatting {
            tactic: ListTactic::Mixed,
            separator: ",",
            trailing_separator: SeparatorTactic::Never,
            indent: indent,
            // 2 = } + ;
            h_width: IDEAL_WIDTH - (indent + path_str.len() + 2),
            v_width: IDEAL_WIDTH - (indent + path_str.len() + 2),
        };

        // TODO handle any comments inbetween items.
        // If `self` is in the list, put it first.
        let head = if path_list.iter().any(|vpi|
            if let ast::PathListItem_::PathListMod{ .. } = vpi.node {
                true
            } else {
                false
            }
        ) {
            Some(("self".to_string(), String::new()))
        } else {
            None
        };

        let items: Vec<_> = head.into_iter().chain(path_list.iter().filter_map(|vpi| {
            match vpi.node {
                ast::PathListItem_::PathListIdent{ name, .. } => {
                    Some((token::get_ident(name).to_string(), String::new()))
                }
                // Skip `self`, because we added it above.
                ast::PathListItem_::PathListMod{ .. } => None,
            }
        })).collect();

        format!("use {}::{{{}}};", path_str, write_list(&items, &fmt))
    }

    fn rewrite_fn(&mut self,
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
        let where_clause = &generics.where_clause;
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

        let ret_str = match fd.output {
            ast::FunctionRetTy::DefaultReturn(_) => String::new(),
            ast::FunctionRetTy::NoReturn(_) => "-> !".to_string(),
            ast::FunctionRetTy::Return(ref ty) => "-> ".to_string() + &pprust::ty_to_string(ty),
        };

        // Args.
        let args = &fd.inputs;

        let mut budgets = None;

        // Try keeping everything on the same line
        if !result.contains("\n") {
            // 3 = `() `, space is before ret_string
            let used_space = indent + result.len() + 3 + ret_str.len();
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
                    arg_strs[0].0 = format!("&{}{}self", lt_str, mut_str)
                }
                ast::ExplicitSelf_::SelfExplicit(ref ty, _) => {
                    arg_strs[0].0 = format!("self: {}", pprust::ty_to_string(ty))
                }
                _ => {}
            }
        }
        result.push_str(&write_list(&arg_strs, &fmt));

        result.push(')');

        // Where clause.
        if where_clause.predicates.len() > 0 {
            result.push('\n');
            result.push_str(&make_indent(indent + 4));
            result.push_str("where ");

            let budget = IDEAL_WIDTH + LEEWAY - indent - 10;
            let fmt = ListFormatting {
                tactic: ListTactic::Vertical,
                separator: ",",
                trailing_separator: SeparatorTactic::Always,
                indent: indent + 10,
                h_width: budget,
                v_width: budget,
            };
            let where_strs: Vec<_> = where_clause.predicates.iter().map(|p| (self.rewrite_pred(p), String::new())).collect();
            result.push_str(&write_list(&where_strs, &fmt));
        }

        // Return type.
        if ret_str.len() > 0 {
            // If we've already gone multi-line, or the return type would push
            // over the max width, then put the return type on a new line.
            if result.contains("\n") ||
               result.len() + indent + ret_str.len() > MAX_WIDTH {
                let indent = indent + 4;
                result.push('\n');
                result.push_str(&make_indent(indent));
            } else {
                result.push(' ');
            }
            result.push_str(&ret_str);
        }

        result
    }

    // TODO we farm this out, but this could spill over the column limit, so we ought to handle it properly
    fn rewrite_fn_input(&self, arg: &ast::Arg) -> String {
        format!("{}: {}",
                pprust::pat_to_string(&arg.pat),
                pprust::ty_to_string(&arg.ty))
    }

    fn rewrite_pred(&self, predicate: &ast::WherePredicate) -> String
    {
        // TODO dead spans
        // TODO assumes we'll always fit on one line...
        match predicate {
            &ast::WherePredicate::BoundPredicate(ast::WhereBoundPredicate{ref bound_lifetimes,
                                                                          ref bounded_ty,
                                                                          ref bounds,
                                                                          ..}) => {
                if bound_lifetimes.len() > 0 {
                    format!("for<{}> {}: {}",
                            bound_lifetimes.iter().map(|l| self.rewrite_lifetime_def(l)).collect::<Vec<_>>().connect(", "),
                            pprust::ty_to_string(bounded_ty),
                            bounds.iter().map(|b| self.rewrite_ty_bound(b)).collect::<Vec<_>>().connect("+"))

                } else {
                    format!("{}: {}",
                            pprust::ty_to_string(bounded_ty),
                            bounds.iter().map(|b| self.rewrite_ty_bound(b)).collect::<Vec<_>>().connect("+"))
                }
            }
            &ast::WherePredicate::RegionPredicate(ast::WhereRegionPredicate{ref lifetime,
                                                                            ref bounds,
                                                                            ..}) => {
                format!("{}: {}",
                        pprust::lifetime_to_string(lifetime),
                        bounds.iter().map(|l| pprust::lifetime_to_string(l)).collect::<Vec<_>>().connect("+"))
            }
            &ast::WherePredicate::EqPredicate(ast::WhereEqPredicate{ref path, ref ty, ..}) => {
                format!("{} = {}", pprust::path_to_string(path), pprust::ty_to_string(ty))
            }
        }
    }

    fn rewrite_lifetime_def(&self, lifetime: &ast::LifetimeDef) -> String
    {
        if lifetime.bounds.len() == 0 {
            return pprust::lifetime_to_string(&lifetime.lifetime);
        }

        format!("{}: {}",
                pprust::lifetime_to_string(&lifetime.lifetime),
                lifetime.bounds.iter().map(|l| pprust::lifetime_to_string(l)).collect::<Vec<_>>().connect("+"))
    }

    fn rewrite_ty_bound(&self, bound: &ast::TyParamBound) -> String
    {
        match *bound {
            ast::TyParamBound::TraitTyParamBound(ref tref, ast::TraitBoundModifier::None) => {
                self.rewrite_poly_trait_ref(tref)
            }
            ast::TyParamBound::TraitTyParamBound(ref tref, ast::TraitBoundModifier::Maybe) => {
                format!("?{}", self.rewrite_poly_trait_ref(tref))
            }
            ast::TyParamBound::RegionTyParamBound(ref l) => {
                pprust::lifetime_to_string(l)
            }
        }
    }

    fn rewrite_ty_param(&self, ty_param: &ast::TyParam) -> String
    {
        let mut result = String::with_capacity(128);
        result.push_str(&token::get_ident(ty_param.ident));
        if ty_param.bounds.len() > 0 {
            result.push_str(": ");
            result.push_str(&ty_param.bounds.iter().map(|b| self.rewrite_ty_bound(b)).collect::<Vec<_>>().connect(", "));
        }
        if let Some(ref def) = ty_param.default {
            result.push_str(" = ");
            result.push_str(&pprust::ty_to_string(&def));
        }

        result
    }

    fn rewrite_poly_trait_ref(&self, t: &ast::PolyTraitRef) -> String
    {
        if t.bound_lifetimes.len() > 0 {
            format!("for<{}> {}",
                    t.bound_lifetimes.iter().map(|l| self.rewrite_lifetime_def(l)).collect::<Vec<_>>().connect(", "),
                    pprust::path_to_string(&t.trait_ref.path))

        } else {
            pprust::path_to_string(&t.trait_ref.path)
        }
    }

    fn rewrite_call(&mut self,
                    callee: &ast::Expr,
                    args: &[ptr::P<ast::Expr>],
                    width: usize,
                    offset: usize)
        -> String
    {
        debug!("rewrite_call, width: {}, offset: {}", width, offset);

        // TODO using byte lens instead of char lens (and probably all over the place too)
        let callee_str = self.rewrite_expr(callee, width, offset);
        debug!("rewrite_call, callee_str: `{}`", callee_str);
        // 2 is for parens.
        let remaining_width = width - callee_str.len() - 2;
        let offset = callee_str.len() + 1 + offset;
        let arg_count = args.len();

        let args_str = if arg_count > 0 {
            let args: Vec<_> = args.iter().map(|e| (self.rewrite_expr(e,
                                                                      remaining_width,
                                                                      offset), String::new())).collect();
            // TODO move this into write_list
            let tactics = if args.iter().any(|&(ref s, _)| s.contains('\n')) {
                ListTactic::Vertical
            } else {
                ListTactic::HorizontalVertical
            };
            let fmt = ListFormatting {
                tactic: tactics,
                separator: ",",
                trailing_separator: SeparatorTactic::Never,
                indent: offset,
                h_width: remaining_width,
                v_width: remaining_width,
            };
            write_list(&args, &fmt)
        } else {
            String::new()
        };

        format!("{}({})", callee_str, args_str)
    }

    fn rewrite_expr(&mut self, expr: &ast::Expr, width: usize, offset: usize) -> String {
        match expr.node {
            ast::Expr_::ExprLit(ref l) => {
                match l.node {
                    ast::Lit_::LitStr(ref is, _) => {
                        return self.rewrite_string_lit(&is, l.span, width, offset);
                    }
                    _ => {}
                }
            }
            ast::Expr_::ExprCall(ref callee, ref args) => {
                return self.rewrite_call(callee, args, width, offset);
            }
            _ => {}
        }

        let result = self.snippet(expr.span);
        debug!("snippet: {}", result);
        result
    }
}

#[inline]
fn prev_char(s: &str, mut i: usize) -> usize {
    if i == 0 { return 0; }

    i -= 1;
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

#[inline]
fn next_char(s: &str, mut i: usize) -> usize {
    if i >= s.len() { return s.len(); }

    while !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

struct RustFmtCalls {
    input_path: Option<PathBuf>,
}

impl<'a> CompilerCalls<'a> for RustFmtCalls {
    fn early_callback(&mut self,
                      _: &getopts::Matches,
                      _: &diagnostics::registry::Registry)
                      -> Compilation {
        Compilation::Continue
    }

    fn some_input(&mut self, input: Input, input_path: Option<PathBuf>) -> (Input, Option<PathBuf>) {
        match input_path {
            Some(ref ip) => self.input_path = Some(ip.clone()),
            _ => {
                // FIXME should handle string input and write to stdout or something
                panic!("No input path");
            }
        }
        (input, input_path)
    }

    fn no_input(&mut self,
                _: &getopts::Matches,
                _: &config::Options,
                _: &Option<PathBuf>,
                _: &Option<PathBuf>,
                _: &diagnostics::registry::Registry)
                -> Option<(Input, Option<PathBuf>)> {
        panic!("No input supplied to RustFmt");
    }

    fn late_callback(&mut self,
                     _: &getopts::Matches,
                     _: &Session,
                     _: &Input,
                     _: &Option<PathBuf>,
                     _: &Option<PathBuf>)
                     -> Compilation {
        Compilation::Continue
    }

    fn build_controller(&mut self, _: &Session) -> driver::CompileController<'a> {
        let mut control = driver::CompileController::basic();
        control.after_parse.stop = Compilation::Stop;
        control.after_parse.callback = box |state| {
            let krate = state.krate.unwrap();
            let codemap = state.session.codemap();
            let mut changes = fmt_ast(krate, codemap);
            fmt_lines(&mut changes);

            println!("{}", changes);
            // FIXME(#5) Should be user specified whether to show or replace.
        };

        control
    }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let mut call_ctxt = RustFmtCalls { input_path: None };
    rustc_driver::run_compiler(&args, &mut call_ctxt);
    std::env::set_exit_status(0);

    // TODO unit tests
    // let fmt = ListFormatting {
    //     tactic: ListTactic::Horizontal,
    //     separator: ",",
    //     trailing_separator: SeparatorTactic::Vertical,
    //     indent: 2,
    //     h_width: 80,
    //     v_width: 100,
    // };
    // let inputs = vec![(format!("foo"), String::new()),
    //                   (format!("foo"), String::new()),
    //                   (format!("foo"), String::new()),
    //                   (format!("foo"), String::new()),
    //                   (format!("foo"), String::new()),
    //                   (format!("foo"), String::new()),
    //                   (format!("foo"), String::new()),
    //                   (format!("foo"), String::new())];
    // let s = write_list(&inputs, &fmt);
    // println!("  {}", s);
}

// FIXME comments
// comments aren't in the AST, which makes processing them difficult, but then
// comments are complicated anyway. I think I am happy putting off tackling them
// for now. Long term the soluton is for comments to be in the AST, but that means
// only the libsyntax AST, not the rustc one, which means waiting for the ASTs
// to diverge one day....

// Once we do have comments, we just have to implement a simple word wrapping
// algorithm to keep the width under IDEAL_WIDTH. We should also convert multiline
// /* ... */ comments to // and check doc comments are in the right place and of
// the right kind.

// Should also make sure comments have the right indent
