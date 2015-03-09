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
#![feature(os)]
#![feature(core)]
#![feature(unicode)]
#![feature(old_path)]
#![feature(exit_status)]

// TODO we're going to allocate a whole bunch of temp Strings, is it worth
// keeping some scratch mem for this and running our own StrPool?

#[macro_use]
extern crate log;

extern crate getopts;
extern crate rustc;
extern crate rustc_driver;
extern crate syntax;

use rustc::session::Session;
use rustc::session::config::{self, Input};
use rustc_driver::{driver, CompilerCalls, Compilation};

use syntax::{ast, ptr};
use syntax::codemap::{self, CodeMap, Span, Pos, BytePos};
use syntax::diagnostics;
use syntax::parse::token;
use syntax::print::pprust;
use syntax::visit;

use std::slice::SliceConcatExt;

use changes::ChangeSet;

pub mod rope;
pub mod string_buffer;
mod changes;

const IDEAL_WIDTH: usize = 80;
const LEEWAY: usize = 5;
const MAX_WIDTH: usize = 100;
const MIN_STRING: usize = 10;

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
        let mut last_wspace = None;
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
    block_indent: usize,
}

impl<'a, 'v> visit::Visitor<'v> for FmtVisitor<'a> {
    fn visit_expr(&mut self, ex: &'v ast::Expr) {
        self.format_missing(ex.span.lo);
        let offset = self.changes.cur_offset_span(ex.span);
        let new_str = self.rewrite_expr(ex, MAX_WIDTH - offset, offset);
        self.changes.push_str_span(ex.span, &new_str);
        self.last_pos = ex.span.hi;
    }

    fn visit_block(&mut self, b: &'v ast::Block) {
        self.format_missing(b.span.lo);

        self.changes.push_str_span(b.span, "{");
        self.last_pos = self.last_pos + BytePos(1);
        self.block_indent += 4;

        for stmt in &b.stmts {
            self.format_missing_with_indent(stmt.span.lo);
            self.visit_stmt(&**stmt)
        }
        match b.expr {
            Some(ref e) => {
                self.format_missing_with_indent(e.span.lo);
                self.visit_expr(e);
            }
            None => {}
        }

        self.block_indent -= 4;
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
        if let Some(new_str) = self.formal_args(fk, fd) {
            self.changes.push_str_span(s, &new_str);            
        }
        visit::walk_fn(self, fk, fd, b, s);
    }

    fn visit_item(&mut self, item: &'v ast::Item) {
        match item.node {
            ast::Item_::ItemUse(ref vp) => {
                match vp.node {
                    ast::ViewPath_::ViewPathList(ref path, ref path_list) => {
                        self.format_missing(item.span.lo);
                        let new_str = self.fix_use_list(path, path_list, vp.span);
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
                self.block_indent += 4;
                visit::walk_item(self, item);
                self.block_indent -= 4;
            }
            _ => {
                visit::walk_item(self, item);
            }
        }
    }
}

fn make_indent(width: usize) -> String {
    let mut indent = String::with_capacity(width);
    for _ in 0..width {
        indent.push(' ')
    }
    indent
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
        // TODO(#11) gets tricky if we're missing more than one file
        assert!(self.codemap.lookup_char_pos(start).file.name == self.codemap.lookup_char_pos(end).file.name,
                "not implemented: unformated span across files");

        self.last_pos = end;
        let span = codemap::mk_sp(start, end);
        let snippet = self.snippet(span);

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
    fn rewrite_string(&mut self, s: &str, span: Span, width: usize, offset: usize) -> String {
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
    fn fix_use_list(&mut self,
                    path: &ast::Path,
                    path_list: &[ast::PathListItem],
                    vp_span: Span) -> String {
        // FIXME remove unused imports

        // FIXME check indentation
        let l_loc = self.codemap.lookup_char_pos(vp_span.lo);
        let path_str = pprust::path_to_string(&path);
        let indent = l_loc.col.0;
        // After accounting for the overhead, how much space left for
        // the item list? ( 5 = :: + { + } + ; )
        let space = IDEAL_WIDTH - (indent + path_str.len() + 5);
        // 4 = `use` + one space
        // TODO might be pub use
        let indent = make_indent(indent-4);

        let mut cur_str = String::new();
        let mut first = true;
        // If `self` is in the list, put it first.
        if path_list.iter().any(|vpi|
            if let ast::PathListItem_::PathListMod{ .. } = vpi.node {
                true
            } else {
                false
            }
        ) {
            cur_str = "self".to_string();
            first = false;
        }

        let mut new_str = String::new();
        for vpi in path_list.iter() {
            match vpi.node {
                ast::PathListItem_::PathListIdent{ name, .. } => {
                    let next_item = &token::get_ident(name);
                    if cur_str.len() + next_item.len() > space {
                        let cur_line = format!("{}use {}::{{{}}};\n", indent, path_str, cur_str);
                        new_str.push_str(&cur_line);

                        cur_str = String::new();
                        first = true;
                    }

                    if first {
                        first = false;
                    } else {
                        cur_str.push_str(", ");
                    }

                    cur_str.push_str(next_item);
                }
                ast::PathListItem_::PathListMod{ .. } => {}
            }
        }

        assert!(!first);
        let cur_line = format!("{}use {}::{{{}}};", indent, path_str, cur_str);
        new_str.push_str(&cur_line);

        new_str
    }

    fn formal_args<'v>(&mut self, fk: visit::FnKind<'v>, fd: &'v ast::FnDecl) -> Option<String> {
        // For now, just check the arguments line up and make them per-row if the line is too long.
        let args = &fd.inputs;

        let ret_str = match fd.output {
            ast::FunctionRetTy::DefaultReturn(_) => "".to_string(),
            ast::FunctionRetTy::NoReturn(_) => " -> !".to_string(),
            ast::FunctionRetTy::Return(ref ty) => pprust::ty_to_string(ty),
        };

        // TODO don't return, want to do the return type etc.
        if args.len() == 0 {
            return None;
        }

        // TODO not really using the hi positions
        let spans: Vec<_> = args.iter().map(|a| (a.pat.span.lo, a.ty.span.hi)).collect();
        let locs: Vec<_> = spans.iter().map(|&(a, b)| {
            (self.codemap.lookup_char_pos(a), self.codemap.lookup_char_pos(b))
        }).collect();
        let first_col = locs[0].0.col.0;

        // Print up to the start of the args.
        self.format_missing(spans[0].0);
        self.last_pos = spans.last().unwrap().1;

        let arg_strs: Vec<_> = args.iter().map(|a| format!("{}: {}",
                                                           pprust::pat_to_string(&a.pat),
                                                           pprust::ty_to_string(&a.ty))).collect();

        // Try putting everything on one row:
        let mut len = arg_strs.iter().fold(0, |a, b| a + b.len());
        // Account for punctuation and spacing.
        len += 2 * arg_strs.len() + 2 * (arg_strs.len()-1);
        // Return type.
        len += ret_str.len();
        // Opening brace if no where clause.
        match fk {
            visit::FnKind::FkItemFn(_, g, _, _) |
            visit::FnKind::FkMethod(_, g, _)
            if g.where_clause.predicates.len() > 0 => {}
            _ => len += 2 // ` {`
        }
        len += first_col;

        if len <= IDEAL_WIDTH + LEEWAY || args.len() == 1 {
            // It should all fit on one line.
            return Some(arg_strs.connect(", "));
        } else {
            // TODO multi-line
            let mut indent = String::with_capacity(first_col + 2);
            indent.push_str(",\n");
            for _ in 0..first_col { indent.push(' '); }
            return Some(arg_strs.connect(&indent));
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
        let args: Vec<_> = args.iter().map(|e| self.rewrite_expr(e,
                                                                 remaining_width,
                                                                 offset)).collect();
        debug!("rewrite_call, args: `{}`", args.connect(","));

        let multi_line = args.iter().any(|s| s.contains('\n'));
        let args_width = args.iter().map(|s| s.len()).fold(0, |a, l| a + l);
        let over_wide = args_width + (arg_count - 1) * 2 > remaining_width;
        let args_str = if multi_line || over_wide {
            args.connect(&(",\n".to_string() + &make_indent(offset)))
        } else {
            args.connect(", ")
        };

        format!("{}({})", callee_str, args_str)
    }

    fn rewrite_expr(&mut self, expr: &ast::Expr, width: usize, offset: usize) -> String {
        match expr.node {
            ast::Expr_::ExprLit(ref l) => {
                match l.node {
                    ast::Lit_::LitStr(ref is, _) => {
                        return self.rewrite_string(&is, l.span, width, offset);
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
    input_path: Option<Path>,
}

impl<'a> CompilerCalls<'a> for RustFmtCalls {
    fn early_callback(&mut self,
                      _: &getopts::Matches,
                      _: &diagnostics::registry::Registry)
                      -> Compilation {
        Compilation::Continue
    }

    fn some_input(&mut self, input: Input, input_path: Option<Path>) -> (Input, Option<Path>) {
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
                _: &Option<Path>,
                _: &Option<Path>,
                _: &diagnostics::registry::Registry)
                -> Option<(Input, Option<Path>)> {
        panic!("No input supplied to RustFmt");
    }

    fn late_callback(&mut self,
                     _: &getopts::Matches,
                     _: &Session,
                     _: &Input,
                     _: &Option<Path>,
                     _: &Option<Path>)
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

            // TODO we stop before expansion, but we still seem to get expanded for loops which
            // cause problems - probably a rustc bug
        };

        control
    }
}

fn main() {
    let args = std::os::args();
    let mut call_ctxt = RustFmtCalls { input_path: None };
    rustc_driver::run_compiler(&args, &mut call_ctxt);
    std::env::set_exit_status(0);
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
