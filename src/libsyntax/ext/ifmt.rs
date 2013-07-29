// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast;
use codemap::span;
use ext::base::*;
use ext::base;
use ext::build::AstBuilder;
use parse;
use parse::token;

use std::fmt::ct;
use std::hashmap::{HashMap, HashSet};

#[deriving(Eq)]
enum ArgumentType {
    Unknown,
    Known(@str),
    Unsigned,
    String,
}

struct Context {
    ecx: @ExtCtxt,
    fmtsp: span,

    // Parsed argument expressions and the types that we've found so far for
    // them.
    args: ~[@ast::expr],
    arg_types: ~[Option<ArgumentType>],
    // Parsed named expressions and the types that we've found for them so far
    names: HashMap<@str, @ast::expr>,
    name_types: HashMap<@str, ArgumentType>,

    // Updated as arguments are consumed or methods are entered
    nest_level: uint,
    next_arg: uint,
}

impl Context {
    /// Parses the arguments from the given list of tokens, returning None if
    /// there's a parse error so we can continue parsing other fmt! expressions.
    fn parse_args(&mut self, sp: span,
                  tts: &[ast::token_tree]) -> Option<@ast::expr> {
        let p = parse::new_parser_from_tts(self.ecx.parse_sess(),
                                           self.ecx.cfg(),
                                           tts.to_owned());
        if *p.token == token::EOF {
            self.ecx.span_err(sp, "ifmt! expects at least one argument");
            return None;
        }
        let fmtstr = p.parse_expr();
        let mut named = false;
        while *p.token != token::EOF {
            if !p.eat(&token::COMMA) {
                self.ecx.span_err(sp, "expected token: `,`");
                return None;
            }
            if named || (token::is_ident(p.token) &&
                         p.look_ahead(1, |t| *t == token::EQ)) {
                named = true;
                let ident = match *p.token {
                    token::IDENT(i, _) => {
                        p.bump();
                        i
                    }
                    _ if named => {
                        self.ecx.span_err(*p.span,
                                          "expected ident, positional arguments \
                                           cannot follow named arguments");
                        return None;
                    }
                    _ => {
                        self.ecx.span_err(*p.span,
                                          fmt!("expected ident for named \
                                                argument, but found `%s`",
                                               p.this_token_to_str()));
                        return None;
                    }
                };
                let name = self.ecx.str_of(ident);
                p.expect(&token::EQ);
                let e = p.parse_expr();
                match self.names.find(&name) {
                    None => {}
                    Some(prev) => {
                        self.ecx.span_err(e.span, fmt!("duplicate argument \
                                                        named `%s`", name));
                        self.ecx.parse_sess.span_diagnostic.span_note(
                            prev.span, "previously here");
                        loop
                    }
                }
                self.names.insert(name, e);
            } else {
                self.args.push(p.parse_expr());
                self.arg_types.push(None);
            }
        }
        return Some(fmtstr);
    }

    /// Verifies one piece of a parse string. All errors are not emitted as
    /// fatal so we can continue giving errors about this and possibly other
    /// format strings.
    fn verify_piece(&mut self, p: &ct::Piece) {
        match *p {
            ct::String(*) => {}
            ct::CurrentArgument => {
                if self.nest_level == 0 {
                    self.ecx.span_err(self.fmtsp,
                                      "`#` reference used with nothing to \
                                       reference back to");
                }
            }
            ct::Argument(ref arg) => {
                // argument first (it's first in the format string)
                let pos = match arg.position {
                    ct::ArgumentNext => {
                        let i = self.next_arg;
                        self.next_arg += 1;
                        Left(i)
                    }
                    ct::ArgumentIs(i) => Left(i),
                    ct::ArgumentNamed(s) => Right(s.to_managed()),
                };
                let ty = if arg.format.ty == "" {
                    Unknown
                } else { Known(arg.format.ty.to_managed()) };
                self.verify_arg_type(pos, ty);

                // width/precision next
                self.verify_count(arg.format.width);
                self.verify_count(arg.format.precision);

                // and finally the method being applied
                match arg.method {
                    None => {}
                    Some(ref method) => { self.verify_method(pos, *method); }
                }
            }
        }
    }

    fn verify_pieces(&mut self, pieces: &[ct::Piece]) {
        for pieces.iter().advance |piece| {
            self.verify_piece(piece);
        }
    }

    fn verify_count(&mut self, c: ct::Count) {
        match c {
            ct::CountImplied | ct::CountIs(*) => {}
            ct::CountIsNextParam => {
                self.verify_arg_type(Left(self.next_arg), Unsigned);
                self.next_arg += 1;
            }
            ct::CountIsParam(i) => {
                self.verify_arg_type(Left(i), Unsigned);
            }
        }
    }

    fn verify_method(&mut self, pos: Either<uint, @str>, m: &ct::Method) {
        self.nest_level += 1;
        match *m {
            ct::Plural(_, ref arms, ref default) => {
                let mut seen_cases = HashSet::new();
                self.verify_arg_type(pos, Unsigned);
                for arms.iter().advance |arm| {
                    if !seen_cases.insert(arm.selector) {
                        match arm.selector {
                            Left(name) => {
                                self.ecx.span_err(self.fmtsp,
                                                  fmt!("duplicate selector \
                                                       `%?`", name));
                            }
                            Right(idx) => {
                                self.ecx.span_err(self.fmtsp,
                                                  fmt!("duplicate selector \
                                                       `=%u`", idx));
                            }
                        }
                    }
                    self.verify_pieces(arm.result);
                }
                self.verify_pieces(*default);
            }
            ct::Select(ref arms, ref default) => {
                self.verify_arg_type(pos, String);
                let mut seen_cases = HashSet::new();
                for arms.iter().advance |arm| {
                    if !seen_cases.insert(arm.selector) {
                        self.ecx.span_err(self.fmtsp,
                                          fmt!("duplicate selector `%s`",
                                               arm.selector));
                    } else if arm.selector == "" {
                        self.ecx.span_err(self.fmtsp,
                                          "empty selector in `select`");
                    }
                    self.verify_pieces(arm.result);
                }
                self.verify_pieces(*default);
            }
        }
        self.nest_level -= 1;
    }

    fn verify_arg_type(&mut self, arg: Either<uint, @str>, ty: ArgumentType) {
        match arg {
            Left(arg) => {
                if arg < 0 || self.args.len() <= arg {
                    let msg = fmt!("invalid reference to argument `%u` (there \
                                    are %u arguments)", arg, self.args.len());
                    self.ecx.span_err(self.fmtsp, msg);
                    return;
                }
                self.verify_same(self.args[arg].span, ty, self.arg_types[arg]);
                if ty != Unknown || self.arg_types[arg].is_none() {
                    self.arg_types[arg] = Some(ty);
                }
            }

            Right(name) => {
                let span = match self.names.find(&name) {
                    Some(e) => e.span,
                    None => {
                        let msg = fmt!("There is no argument named `%s`", name);
                        self.ecx.span_err(self.fmtsp, msg);
                        return;
                    }
                };
                self.verify_same(span, ty,
                                 self.name_types.find(&name).map(|&x| *x));
                if ty != Unknown || !self.name_types.contains_key(&name) {
                    self.name_types.insert(name, ty);
                }
            }
        }
    }

    /// When we're keeping track of the types that are declared for certain
    /// arguments, we assume that `None` means we haven't seen this argument
    /// yet, `Some(None)` means that we've seen the argument, but no format was
    /// specified, and `Some(Some(x))` means that the argument was declared to
    /// have type `x`.
    ///
    /// Obviously `Some(Some(x)) != Some(Some(y))`, but we consider it true
    /// that: `Some(None) == Some(Some(x))`
    fn verify_same(&self, sp: span, ty: ArgumentType,
                   before: Option<ArgumentType>) {
        if ty == Unknown { return }
        let cur = match before {
            Some(Unknown) | None => return,
            Some(t) => t,
        };
        if ty == cur { return }
        match (cur, ty) {
            (Known(cur), Known(ty)) => {
                self.ecx.span_err(sp,
                                  fmt!("argument redeclared with type `%s` when \
                                        it was previously `%s`", ty, cur));
            }
            (Known(cur), _) => {
                self.ecx.span_err(sp,
                                  fmt!("argument used to format with `%s` was \
                                        attempted to not be used for formatting",
                                        cur));
            }
            (_, Known(ty)) => {
                self.ecx.span_err(sp,
                                  fmt!("argument previously used as a format \
                                        argument attempted to be used as `%s`",
                                        ty));
            }
            (_, _) => {
                self.ecx.span_err(sp, "argument declared with multiple formats");
            }
        }
    }

    /// Actually builds the expression which the ifmt! block will be expanded
    /// to
    fn to_expr(&self, fmtstr: @ast::expr) -> @ast::expr {
        let mut lets = ~[];
        let mut locals = ~[];
        let mut names = ~[];

        // Right now there is a bug such that for the expression:
        //      foo(bar(&1))
        // the lifetime of `1` doesn't outlast the call to `bar`, so it's not
        // vald for the call to `foo`. To work around this all arguments to the
        // fmt! string are shoved into locals.
        for self.args.iter().enumerate().advance |(i, &e)| {
            if self.arg_types[i].is_none() { loop } // error already generated

            let name = self.ecx.ident_of(fmt!("__arg%u", i));
            lets.push(self.ecx.stmt_let(e.span, false, name, e));
            locals.push(self.format_arg(e.span, Left(i), name));
        }
        for self.names.iter().advance |(&name, &e)| {
            if !self.name_types.contains_key(&name) { loop }

            let lname = self.ecx.ident_of(fmt!("__arg%s", name));
            lets.push(self.ecx.stmt_let(e.span, false, lname, e));
            let tup = ~[self.ecx.expr_str(e.span, name),
                        self.format_arg(e.span, Right(name), lname)];
            names.push(self.ecx.expr(e.span, ast::expr_tup(tup)));
        }

        // Next, build up the actual call to the sprintf function. This takes
        // three arguments:
        //   1. The format string
        //   2. An array of arguments
        //   3. An array of (name, argument) pairs
        let result = self.ecx.expr_call_global(self.fmtsp, ~[
                self.ecx.ident_of("std"),
                self.ecx.ident_of("fmt"),
                self.ecx.ident_of("sprintf"),
            ], ~[
                fmtstr,
                self.ecx.expr_vec(fmtstr.span, locals),
                self.ecx.expr_vec(fmtstr.span, names)
            ]);

        // sprintf is unsafe, but we just went through a lot of work to
        // validate that our call is save, so inject the unsafe block for the
        // user.
        let result = self.ecx.expr_blk(ast::Block {
           view_items: ~[],
           stmts: ~[],
           expr: Some(result),
           id: self.ecx.next_id(),
           rules: ast::UnsafeBlock,
           span: self.fmtsp,
        });

        self.ecx.expr_blk(self.ecx.blk(self.fmtsp, lets, Some(result)))
    }

    fn format_arg(&self, sp: span, arg: Either<uint, @str>,
                  ident: ast::ident) -> @ast::expr {
        let mut ty = match arg {
            Left(i) => self.arg_types[i].unwrap(),
            Right(s) => *self.name_types.get(&s)
        };
        // Default types to '?' if nothing else is specified.
        if ty == Unknown {
            ty = Known(@"?");
        }
        let argptr = self.ecx.expr_addr_of(sp, self.ecx.expr_ident(sp, ident));
        match ty {
            Known(tyname) => {
                let format_fn = self.ecx.expr(sp, ast::expr_extfmt_fn(tyname));
                self.ecx.expr_call_global(sp, ~[
                        self.ecx.ident_of("std"),
                        self.ecx.ident_of("fmt"),
                        self.ecx.ident_of("argument"),
                    ], ~[format_fn, argptr])
            }
            String => {
                self.ecx.expr_call_global(sp, ~[
                        self.ecx.ident_of("std"),
                        self.ecx.ident_of("fmt"),
                        self.ecx.ident_of("argumentstr"),
                    ], ~[argptr])
            }
            Unsigned => {
                self.ecx.expr_call_global(sp, ~[
                        self.ecx.ident_of("std"),
                        self.ecx.ident_of("fmt"),
                        self.ecx.ident_of("argumentuint"),
                    ], ~[argptr])
            }
            Unknown => { fail!() }
        }
    }
}

pub fn expand_syntax_ext(ecx: @ExtCtxt, sp: span,
                         tts: &[ast::token_tree]) -> base::MacResult {
    let mut cx = Context {
        ecx: ecx,
        args: ~[],
        arg_types: ~[],
        names: HashMap::new(),
        name_types: HashMap::new(),
        nest_level: 0,
        next_arg: 0,
        fmtsp: sp,
    };
    let efmt = match cx.parse_args(sp, tts) {
        Some(e) => e,
        None => { return MRExpr(ecx.expr_uint(sp, 2)); }
    };
    cx.fmtsp = efmt.span;
    let fmt = expr_to_str(ecx, efmt,
                          ~"first argument to ifmt! must be a string literal.");

    let mut err = false;
    for ct::Parser::new(fmt, |m| {
        if !err {
            err = true;
            ecx.span_err(efmt.span, m);
        }
    }).advance |piece| {
        if !err {
            cx.verify_piece(&piece);
        }
    }
    if err { return MRExpr(efmt); }

    // Make sure that all arguments were used and all arguments have types.
    for cx.arg_types.iter().enumerate().advance |(i, ty)| {
        if ty.is_none() {
            ecx.span_err(cx.args[i].span, "argument never used");
        }
    }
    for cx.names.iter().advance |(name, e)| {
        if !cx.name_types.contains_key(name) {
            ecx.span_err(e.span, "named argument never used");
        }
    }

    MRExpr(cx.to_expr(efmt))
}
