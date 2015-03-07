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

#[macro_use]
extern crate log;

extern crate getopts;
extern crate rustc;
extern crate rustc_driver;
extern crate syntax;

use rustc::session::Session;
use rustc::session::config::{self, Input};
use rustc_driver::{driver, CompilerCalls, Compilation};

use syntax::ast;
use syntax::codemap::{self, CodeMap, Span, Pos};
use syntax::diagnostics;
use syntax::parse::token;
use syntax::print::pprust;
use syntax::visit;

use std::mem;

use changes::ChangeSet;

pub mod rope;
mod changes;

const IDEAL_WIDTH: usize = 80;
const MAX_WIDTH: usize = 100;
const MIN_STRING: usize = 10;

// Formatting which depends on the AST.
fn fmt_ast<'a>(krate: &ast::Crate, codemap: &'a CodeMap) -> ChangeSet<'a> {
    let mut visitor = FmtVisitor { codemap: codemap,
                                   changes: ChangeSet::from_codemap(codemap) };
    visit::walk_crate(&mut visitor, krate);

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
                    trims.push((lw, b));
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

        unsafe {
            // Invariant: we only mutate a rope after we have searched it, then
            // we will not search it again.
            let mut_text: &mut rope::Rope = mem::transmute(text);
            let mut_count: &mut u64 = mem::transmute(&changes.count);
            let mut offset = 0;
            // Get rid of any trailing whitespace we recorded earlier.
            for &(s, e) in trims.iter() {
                // Note that we change the underlying ropes directly, we don't
                // go through the changeset because our change positions are
                // relative to the newest text, not the original.
                debug!("Stripping trailing whitespace {}:{}-{} \"{}\"",
                       f, s, e, text.slice(s-offset..e-offset));
                mut_text.remove(s-offset, e-offset);
                *mut_count += 1;
                offset += e - s;
            }
        }
    }
}

struct FmtVisitor<'a> {
    codemap: &'a CodeMap,
    changes: ChangeSet<'a>,
}

impl<'a, 'v> visit::Visitor<'v> for FmtVisitor<'a> {
    fn visit_expr(&mut self, ex: &'v ast::Expr) {
        match ex.node {
            ast::Expr_::ExprLit(ref l) => match l.node {
                ast::Lit_::LitStr(ref is, _) => {
                    self.rewrite_string(&is, l.span);
                }
                _ => {}
            },
            _ => {}
        }

        visit::walk_expr(self, ex)
    }

    fn visit_fn(&mut self,
                fk: visit::FnKind<'v>,
                fd: &'v ast::FnDecl,
                b: &'v ast::Block,
                s: Span,
                _: ast::NodeId) {
        self.fix_formal_args(fd);
        visit::walk_fn(self, fk, fd, b, s);
    }

    fn visit_item(&mut self, item: &'v ast::Item) {
        // TODO check each item is on a new line and is correctly indented.
        match item.node {
            ast::Item_::ItemUse(ref vp) => {
                match vp.node {
                    ast::ViewPath_::ViewPathList(ref path, ref path_list) => {
                        let new_str = self.fix_use_list(path, path_list, vp.span);

                        // TODO move these optimisations to ChangeSet
                        if new_str != self.codemap.span_to_snippet(item.span).unwrap() {
                            self.changes.change_span(item.span, new_str);
                        }
                    }
                    ast::ViewPath_::ViewPathGlob(_) => {
                        // FIXME convert to list?
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        visit::walk_item(self, item);
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
    // TODO NEEDS TESTS
    fn rewrite_string(&mut self, s: &str, span: Span) {
        // FIXME I bet this stomps unicode escapes in the source string

        // Check if there is anything to fix: we always try to fixup multi-line
        // strings, or if the string is too long for the line.
        let l_loc = self.codemap.lookup_char_pos(span.lo);
        let r_loc = self.codemap.lookup_char_pos(span.hi);
        if l_loc.line == r_loc.line && r_loc.col.to_usize() <= MAX_WIDTH {
            return;
        }

        // TODO if lo.col > IDEAL - 10, start a new line (need cur indent for that)

        let s = s.escape_default();

        // TODO use fixed value.
        let l_loc = self.codemap.lookup_char_pos(span.lo);
        let l_col = l_loc.col.to_usize();
        
        let indent = make_indent(l_col + 1);
        let indent = &indent;

        let max_chars = MAX_WIDTH - (l_col + 1);

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

        // Check that we actually changed something.
        if result == self.codemap.span_to_snippet(span).unwrap() {
            return;
        }

        self.changes.change_span(span, result);
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

    fn fix_formal_args<'v>(&mut self, fd: &'v ast::FnDecl) {
        // For now, just check the arguments line up and make them per-row if the line is too long.
        let args = &fd.inputs;
        if args.len() <= 1 {
            return;
        }
        // TODO not really using the hi positions
        let spans: Vec<_> = args.iter().map(|a| (a.pat.span.lo, a.ty.span.hi)).collect();
        let locs: Vec<_> = spans.iter().map(|&(a, b)| (self.codemap.lookup_char_pos(a), self.codemap.lookup_char_pos(b))).collect();
        let first_loc = &locs[0].0;
        // TODO need to adjust for previous changes here.
        let same_row = locs.iter().all(|&(ref l, _)| l.line == first_loc.line);
        let same_col = locs.iter().all(|&(ref l, _)| l.col == first_loc.col);

        if same_col {
            // TODO Check one arg per line and no lines in between (except comments)
            return;
        }        

        if same_row { // TODO check line is < 100 && first_loc.line {
            // TODO could also fix whitespace around punctuaton here
            // TODO and could check that we're on the same line as the function call, if possible
            return;
        }

        let col = self.changes.col(spans[0].0);
        let mut indent = String::with_capacity(col);
        indent.push('\n');
        for _ in 0..col { indent.push(' '); }
        let last_idx = spans.len() - 1;
        for (i, s) in spans.iter().enumerate() {
            // Take the span from lo to lo (or the last hi for the last arg), 
            // trim, push on top of indent, then replace the old lo-lo span with it.
            let mut new_text = if i == 0 {
                "".to_string()
            } else {
                indent.clone()
            };
            let hi = if i == last_idx {
                s.1
            } else {
                spans[i+1].0
            };
            // TODO need a version of slice taking locs, not a span
            let snippet = self.changes.slice_span(Span{ lo: s.0, hi: hi, expn_id: codemap::NO_EXPANSION }).to_string();
            let snippet = snippet.trim();
            new_text.push_str(snippet);
            self.changes.change(&first_loc.file.name, (s.0).0 as usize, hi.0 as usize, new_text);
        }
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

            println!("Making {} changes", changes.count);
            println!("{}", changes);
            // FIXME(#5) Should be user specified whether to show or replace.
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
