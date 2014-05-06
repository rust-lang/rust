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
use ast::P;
use codemap::{Span, respan};
use ext::base::*;
use ext::base;
use ext::build::AstBuilder;
use parse::token::InternedString;
use parse::token;
use rsparse = parse;

use std::fmt::parse;
use collections::{HashMap, HashSet};

#[deriving(Eq)]
enum ArgumentType {
    Known(~str),
    Unsigned,
    String,
}

enum Position {
    Exact(uint),
    Named(~str),
}

struct Context<'a, 'b> {
    ecx: &'a mut ExtCtxt<'b>,
    fmtsp: Span,

    // Parsed argument expressions and the types that we've found so far for
    // them.
    args: Vec<@ast::Expr>,
    arg_types: Vec<Option<ArgumentType>>,
    // Parsed named expressions and the types that we've found for them so far.
    // Note that we keep a side-array of the ordering of the named arguments
    // found to be sure that we can translate them in the same order that they
    // were declared in.
    names: HashMap<~str, @ast::Expr>,
    name_types: HashMap<~str, ArgumentType>,
    name_ordering: Vec<~str>,

    // Collection of the compiled `rt::Piece` structures
    pieces: Vec<@ast::Expr> ,
    name_positions: HashMap<~str, uint>,
    method_statics: Vec<@ast::Item> ,

    // Updated as arguments are consumed or methods are entered
    nest_level: uint,
    next_arg: uint,
}

/// Parses the arguments from the given list of tokens, returning None
/// if there's a parse error so we can continue parsing other format!
/// expressions.
///
/// If parsing succeeds, the second return value is:
///
///     Some((fmtstr, unnamed arguments, ordering of named arguments,
///           named arguments))
fn parse_args(ecx: &mut ExtCtxt, sp: Span, tts: &[ast::TokenTree])
    -> (@ast::Expr, Option<(@ast::Expr, Vec<@ast::Expr>, Vec<~str>,
                            HashMap<~str, @ast::Expr>)>) {
    let mut args = Vec::new();
    let mut names = HashMap::<~str, @ast::Expr>::new();
    let mut order = Vec::new();

    let mut p = rsparse::new_parser_from_tts(ecx.parse_sess(),
                                             ecx.cfg(),
                                             tts.iter()
                                                .map(|x| (*x).clone())
                                                .collect());
    // Parse the leading function expression (maybe a block, maybe a path)
    let extra = p.parse_expr();
    if !p.eat(&token::COMMA) {
        ecx.span_err(sp, "expected token: `,`");
        return (extra, None);
    }

    if p.token == token::EOF {
        ecx.span_err(sp, "requires at least a format string argument");
        return (extra, None);
    }
    let fmtstr = p.parse_expr();
    let mut named = false;
    while p.token != token::EOF {
        if !p.eat(&token::COMMA) {
            ecx.span_err(sp, "expected token: `,`");
            return (extra, None);
        }
        if p.token == token::EOF { break } // accept trailing commas
        if named || (token::is_ident(&p.token) &&
                     p.look_ahead(1, |t| *t == token::EQ)) {
            named = true;
            let ident = match p.token {
                token::IDENT(i, _) => {
                    p.bump();
                    i
                }
                _ if named => {
                    ecx.span_err(p.span,
                                 "expected ident, positional arguments \
                                 cannot follow named arguments");
                    return (extra, None);
                }
                _ => {
                    ecx.span_err(p.span,
                                 format!("expected ident for named argument, but found `{}`",
                                         p.this_token_to_str()));
                    return (extra, None);
                }
            };
            let interned_name = token::get_ident(ident);
            let name = interned_name.get();
            p.expect(&token::EQ);
            let e = p.parse_expr();
            match names.find_equiv(&name) {
                None => {}
                Some(prev) => {
                    ecx.span_err(e.span, format!("duplicate argument named `{}`", name));
                    ecx.parse_sess.span_diagnostic.span_note(prev.span, "previously here");
                    continue
                }
            }
            order.push(name.to_str());
            names.insert(name.to_str(), e);
        } else {
            args.push(p.parse_expr());
        }
    }
    return (extra, Some((fmtstr, args, order, names)));
}

impl<'a, 'b> Context<'a, 'b> {
    /// Verifies one piece of a parse string. All errors are not emitted as
    /// fatal so we can continue giving errors about this and possibly other
    /// format strings.
    fn verify_piece(&mut self, p: &parse::Piece) {
        match *p {
            parse::String(..) => {}
            parse::CurrentArgument => {
                if self.nest_level == 0 {
                    self.ecx.span_err(self.fmtsp,
                                      "`#` reference used with nothing to \
                                       reference back to");
                }
            }
            parse::Argument(ref arg) => {
                // width/precision first, if they have implicit positional
                // parameters it makes more sense to consume them first.
                self.verify_count(arg.format.width);
                self.verify_count(arg.format.precision);

                // argument second, if it's an implicit positional parameter
                // it's written second, so it should come after width/precision.
                let pos = match arg.position {
                    parse::ArgumentNext => {
                        let i = self.next_arg;
                        if self.check_positional_ok() {
                            self.next_arg += 1;
                        }
                        Exact(i)
                    }
                    parse::ArgumentIs(i) => Exact(i),
                    parse::ArgumentNamed(s) => Named(s.to_str()),
                };

                // and finally the method being applied
                match arg.method {
                    None => {
                        let ty = Known(arg.format.ty.to_str());
                        self.verify_arg_type(pos, ty);
                    }
                    Some(ref method) => { self.verify_method(pos, *method); }
                }
            }
        }
    }

    fn verify_pieces(&mut self, pieces: &[parse::Piece]) {
        for piece in pieces.iter() {
            self.verify_piece(piece);
        }
    }

    fn verify_count(&mut self, c: parse::Count) {
        match c {
            parse::CountImplied | parse::CountIs(..) => {}
            parse::CountIsParam(i) => {
                self.verify_arg_type(Exact(i), Unsigned);
            }
            parse::CountIsName(s) => {
                self.verify_arg_type(Named(s.to_str()), Unsigned);
            }
            parse::CountIsNextParam => {
                if self.check_positional_ok() {
                    self.verify_arg_type(Exact(self.next_arg), Unsigned);
                    self.next_arg += 1;
                }
            }
        }
    }

    fn check_positional_ok(&mut self) -> bool {
        if self.nest_level != 0 {
            self.ecx.span_err(self.fmtsp, "cannot use implicit positional \
                                           arguments nested inside methods");
            false
        } else {
            true
        }
    }

    fn verify_method(&mut self, pos: Position, m: &parse::Method) {
        self.nest_level += 1;
        match *m {
            parse::Plural(_, ref arms, ref default) => {
                let mut seen_cases = HashSet::new();
                self.verify_arg_type(pos, Unsigned);
                for arm in arms.iter() {
                    if !seen_cases.insert(arm.selector) {
                        match arm.selector {
                            parse::Keyword(name) => {
                                self.ecx.span_err(self.fmtsp,
                                                  format!("duplicate selector \
                                                           `{:?}`", name));
                            }
                            parse::Literal(idx) => {
                                self.ecx.span_err(self.fmtsp,
                                                  format!("duplicate selector \
                                                           `={}`", idx));
                            }
                        }
                    }
                    self.verify_pieces(arm.result.as_slice());
                }
                self.verify_pieces(default.as_slice());
            }
            parse::Select(ref arms, ref default) => {
                self.verify_arg_type(pos, String);
                let mut seen_cases = HashSet::new();
                for arm in arms.iter() {
                    if !seen_cases.insert(arm.selector) {
                        self.ecx.span_err(self.fmtsp,
                                          format!("duplicate selector `{}`",
                                               arm.selector));
                    } else if arm.selector == "" {
                        self.ecx.span_err(self.fmtsp,
                                          "empty selector in `select`");
                    }
                    self.verify_pieces(arm.result.as_slice());
                }
                self.verify_pieces(default.as_slice());
            }
        }
        self.nest_level -= 1;
    }

    fn verify_arg_type(&mut self, arg: Position, ty: ArgumentType) {
        match arg {
            Exact(arg) => {
                if self.args.len() <= arg {
                    let msg = format!("invalid reference to argument `{}` (there \
                                    are {} arguments)", arg, self.args.len());
                    self.ecx.span_err(self.fmtsp, msg);
                    return;
                }
                {
                    let arg_type = match self.arg_types.get(arg) {
                        &None => None,
                        &Some(ref x) => Some(x)
                    };
                    self.verify_same(self.args.get(arg).span, &ty, arg_type);
                }
                if self.arg_types.get(arg).is_none() {
                    *self.arg_types.get_mut(arg) = Some(ty);
                }
            }

            Named(name) => {
                let span = match self.names.find(&name) {
                    Some(e) => e.span,
                    None => {
                        let msg = format!("there is no argument named `{}`", name);
                        self.ecx.span_err(self.fmtsp, msg);
                        return;
                    }
                };
                self.verify_same(span, &ty, self.name_types.find(&name));
                if !self.name_types.contains_key(&name) {
                    self.name_types.insert(name.clone(), ty);
                }
                // Assign this named argument a slot in the arguments array if
                // it hasn't already been assigned a slot.
                if !self.name_positions.contains_key(&name) {
                    let slot = self.name_positions.len();
                    self.name_positions.insert(name, slot);
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
    fn verify_same(&self,
                   sp: Span,
                   ty: &ArgumentType,
                   before: Option<&ArgumentType>) {
        let cur = match before {
            None => return,
            Some(t) => t,
        };
        if *ty == *cur {
            return
        }
        match (cur, ty) {
            (&Known(ref cur), &Known(ref ty)) => {
                self.ecx.span_err(sp,
                                  format!("argument redeclared with type `{}` when \
                                           it was previously `{}`",
                                          *ty,
                                          *cur));
            }
            (&Known(ref cur), _) => {
                self.ecx.span_err(sp,
                                  format!("argument used to format with `{}` was \
                                           attempted to not be used for formatting",
                                           *cur));
            }
            (_, &Known(ref ty)) => {
                self.ecx.span_err(sp,
                                  format!("argument previously used as a format \
                                           argument attempted to be used as `{}`",
                                           *ty));
            }
            (_, _) => {
                self.ecx.span_err(sp, "argument declared with multiple formats");
            }
        }
    }

    /// These attributes are applied to all statics that this syntax extension
    /// will generate.
    fn static_attrs(&self) -> Vec<ast::Attribute> {
        // Flag statics as `address_insignificant` so LLVM can merge duplicate
        // globals as much as possible (which we're generating a whole lot of).
        let unnamed = self.ecx
                          .meta_word(self.fmtsp,
                                     InternedString::new(
                                         "address_insignificant"));
        let unnamed = self.ecx.attribute(self.fmtsp, unnamed);

        // Do not warn format string as dead code
        let dead_code = self.ecx.meta_word(self.fmtsp,
                                           InternedString::new("dead_code"));
        let allow_dead_code = self.ecx.meta_list(self.fmtsp,
                                                 InternedString::new("allow"),
                                                 vec!(dead_code));
        let allow_dead_code = self.ecx.attribute(self.fmtsp, allow_dead_code);
        return vec!(unnamed, allow_dead_code);
    }

    fn parsepath(&self, s: &str) -> Vec<ast::Ident> {
        vec!(self.ecx.ident_of("std"), self.ecx.ident_of("fmt"),
          self.ecx.ident_of("parse"), self.ecx.ident_of(s))
    }

    fn rtpath(&self, s: &str) -> Vec<ast::Ident> {
        vec!(self.ecx.ident_of("std"), self.ecx.ident_of("fmt"),
          self.ecx.ident_of("rt"), self.ecx.ident_of(s))
    }

    fn ctpath(&self, s: &str) -> Vec<ast::Ident> {
        vec!(self.ecx.ident_of("std"), self.ecx.ident_of("fmt"),
          self.ecx.ident_of("parse"), self.ecx.ident_of(s))
    }

    fn none(&self) -> @ast::Expr {
        let none = self.ecx.path_global(self.fmtsp, vec!(
                self.ecx.ident_of("std"),
                self.ecx.ident_of("option"),
                self.ecx.ident_of("None")));
        self.ecx.expr_path(none)
    }

    fn some(&self, e: @ast::Expr) -> @ast::Expr {
        let p = self.ecx.path_global(self.fmtsp, vec!(
                self.ecx.ident_of("std"),
                self.ecx.ident_of("option"),
                self.ecx.ident_of("Some")));
        let p = self.ecx.expr_path(p);
        self.ecx.expr_call(self.fmtsp, p, vec!(e))
    }

    fn trans_count(&self, c: parse::Count) -> @ast::Expr {
        let sp = self.fmtsp;
        match c {
            parse::CountIs(i) => {
                self.ecx.expr_call_global(sp, self.rtpath("CountIs"),
                                          vec!(self.ecx.expr_uint(sp, i)))
            }
            parse::CountIsParam(i) => {
                self.ecx.expr_call_global(sp, self.rtpath("CountIsParam"),
                                          vec!(self.ecx.expr_uint(sp, i)))
            }
            parse::CountImplied => {
                let path = self.ecx.path_global(sp, self.rtpath("CountImplied"));
                self.ecx.expr_path(path)
            }
            parse::CountIsNextParam => {
                let path = self.ecx.path_global(sp, self.rtpath("CountIsNextParam"));
                self.ecx.expr_path(path)
            }
            parse::CountIsName(n) => {
                let i = match self.name_positions.find_equiv(&n) {
                    Some(&i) => i,
                    None => 0, // error already emitted elsewhere
                };
                let i = i + self.args.len();
                self.ecx.expr_call_global(sp, self.rtpath("CountIsParam"),
                                          vec!(self.ecx.expr_uint(sp, i)))
            }
        }
    }

    fn trans_method(&mut self, method: &parse::Method) -> @ast::Expr {
        let sp = self.fmtsp;
        let method = match *method {
            parse::Select(ref arms, ref default) => {
                let arms = arms.iter().map(|arm| {
                        let p = self.ecx.path_global(sp, self.rtpath("SelectArm"));
                        let result = arm.result.iter().map(|p| {
                            self.trans_piece(p)
                        }).collect();
                        let s = token::intern_and_get_ident(arm.selector);
                        let selector = self.ecx.expr_str(sp, s);
                        self.ecx.expr_struct(sp, p, vec!(
                                self.ecx.field_imm(sp,
                                                   self.ecx.ident_of("selector"),
                                                   selector),
                                self.ecx.field_imm(sp, self.ecx.ident_of("result"),
                                                   self.ecx.expr_vec_slice(sp, result))))
                    }).collect();
                let default = default.iter().map(|p| {
                        self.trans_piece(p)
                    }).collect();
                self.ecx.expr_call_global(sp, self.rtpath("Select"), vec!(
                        self.ecx.expr_vec_slice(sp, arms),
                        self.ecx.expr_vec_slice(sp, default)))
            }
            parse::Plural(offset, ref arms, ref default) => {
                let offset = match offset {
                    Some(i) => { self.some(self.ecx.expr_uint(sp, i)) }
                    None => { self.none() }
                };
                let arms = arms.iter().map(|arm| {
                        let p = self.ecx.path_global(sp, self.rtpath("PluralArm"));
                        let result = arm.result.iter().map(|p| {
                                self.trans_piece(p)
                            }).collect();
                        let (lr, selarg) = match arm.selector {
                            parse::Keyword(t) => {
                                let p = self.ctpath(format!("{:?}", t));
                                let p = self.ecx.path_global(sp, p);
                                (self.rtpath("Keyword"), self.ecx.expr_path(p))
                            }
                            parse::Literal(i) => {
                                (self.rtpath("Literal"), self.ecx.expr_uint(sp, i))
                            }
                        };
                        let selector = self.ecx.expr_call_global(sp,
                                                                 lr, vec!(selarg));
                        self.ecx.expr_struct(sp, p, vec!(
                                self.ecx.field_imm(sp,
                                                   self.ecx.ident_of("selector"),
                                                   selector),
                                self.ecx.field_imm(sp, self.ecx.ident_of("result"),
                                                   self.ecx.expr_vec_slice(sp, result))))
                    }).collect();
                let default = default.iter().map(|p| {
                        self.trans_piece(p)
                    }).collect();
                self.ecx.expr_call_global(sp, self.rtpath("Plural"), vec!(
                        offset,
                        self.ecx.expr_vec_slice(sp, arms),
                        self.ecx.expr_vec_slice(sp, default)))
            }
        };
        let life = self.ecx.lifetime(sp, self.ecx.ident_of("static").name);
        let ty = self.ecx.ty_path(self.ecx.path_all(
                sp,
                true,
                self.rtpath("Method"),
                vec!(life),
                Vec::new()
                    ), None);
        let st = ast::ItemStatic(ty, ast::MutImmutable, method);
        let static_name = self.ecx.ident_of(format!("__STATIC_METHOD_{}",
                                                    self.method_statics.len()));
        let item = self.ecx.item(sp, static_name, self.static_attrs(), st);
        self.method_statics.push(item);
        self.ecx.expr_ident(sp, static_name)
    }

    /// Translate a `parse::Piece` to a static `rt::Piece`
    fn trans_piece(&mut self, piece: &parse::Piece) -> @ast::Expr {
        let sp = self.fmtsp;
        match *piece {
            parse::String(s) => {
                let s = token::intern_and_get_ident(s);
                self.ecx.expr_call_global(sp,
                                          self.rtpath("String"),
                                          vec!(
                    self.ecx.expr_str(sp, s)
                ))
            }
            parse::CurrentArgument => {
                let nil = self.ecx.expr_lit(sp, ast::LitNil);
                self.ecx.expr_call_global(sp, self.rtpath("CurrentArgument"), vec!(nil))
            }
            parse::Argument(ref arg) => {
                // Translate the position
                let pos = match arg.position {
                    // These two have a direct mapping
                    parse::ArgumentNext => {
                        let path = self.ecx.path_global(sp,
                                                        self.rtpath("ArgumentNext"));
                        self.ecx.expr_path(path)
                    }
                    parse::ArgumentIs(i) => {
                        self.ecx.expr_call_global(sp, self.rtpath("ArgumentIs"),
                                                  vec!(self.ecx.expr_uint(sp, i)))
                    }
                    // Named arguments are converted to positional arguments at
                    // the end of the list of arguments
                    parse::ArgumentNamed(n) => {
                        let i = match self.name_positions.find_equiv(&n) {
                            Some(&i) => i,
                            None => 0, // error already emitted elsewhere
                        };
                        let i = i + self.args.len();
                        self.ecx.expr_call_global(sp, self.rtpath("ArgumentIs"),
                                                  vec!(self.ecx.expr_uint(sp, i)))
                    }
                };

                // Translate the format
                let fill = match arg.format.fill { Some(c) => c, None => ' ' };
                let fill = self.ecx.expr_lit(sp, ast::LitChar(fill));
                let align = match arg.format.align {
                    parse::AlignLeft => {
                        self.ecx.path_global(sp, self.parsepath("AlignLeft"))
                    }
                    parse::AlignRight => {
                        self.ecx.path_global(sp, self.parsepath("AlignRight"))
                    }
                    parse::AlignUnknown => {
                        self.ecx.path_global(sp, self.parsepath("AlignUnknown"))
                    }
                };
                let align = self.ecx.expr_path(align);
                let flags = self.ecx.expr_uint(sp, arg.format.flags);
                let prec = self.trans_count(arg.format.precision);
                let width = self.trans_count(arg.format.width);
                let path = self.ecx.path_global(sp, self.rtpath("FormatSpec"));
                let fmt = self.ecx.expr_struct(sp, path, vec!(
                    self.ecx.field_imm(sp, self.ecx.ident_of("fill"), fill),
                    self.ecx.field_imm(sp, self.ecx.ident_of("align"), align),
                    self.ecx.field_imm(sp, self.ecx.ident_of("flags"), flags),
                    self.ecx.field_imm(sp, self.ecx.ident_of("precision"), prec),
                    self.ecx.field_imm(sp, self.ecx.ident_of("width"), width)));

                // Translate the method (if any)
                let method = match arg.method {
                    None => { self.none() }
                    Some(ref m) => {
                        let m = self.trans_method(*m);
                        self.some(self.ecx.expr_addr_of(sp, m))
                    }
                };
                let path = self.ecx.path_global(sp, self.rtpath("Argument"));
                let s = self.ecx.expr_struct(sp, path, vec!(
                    self.ecx.field_imm(sp, self.ecx.ident_of("position"), pos),
                    self.ecx.field_imm(sp, self.ecx.ident_of("format"), fmt),
                    self.ecx.field_imm(sp, self.ecx.ident_of("method"), method)));
                self.ecx.expr_call_global(sp, self.rtpath("Argument"), vec!(s))
            }
        }
    }

    /// Actually builds the expression which the iformat! block will be expanded
    /// to
    fn to_expr(&self, extra: @ast::Expr) -> @ast::Expr {
        let mut lets = Vec::new();
        let mut locals = Vec::new();
        let mut names = Vec::from_fn(self.name_positions.len(), |_| None);
        let mut pats = Vec::new();
        let mut heads = Vec::new();

        // First, declare all of our methods that are statics
        for &method in self.method_statics.iter() {
            let decl = respan(self.fmtsp, ast::DeclItem(method));
            lets.push(@respan(self.fmtsp,
                              ast::StmtDecl(@decl, ast::DUMMY_NODE_ID)));
        }

        // Next, build up the static array which will become our precompiled
        // format "string"
        let fmt = self.ecx.expr_vec(self.fmtsp, self.pieces.clone());
        let piece_ty = self.ecx.ty_path(self.ecx.path_all(
                self.fmtsp,
                true, vec!(
                    self.ecx.ident_of("std"),
                    self.ecx.ident_of("fmt"),
                    self.ecx.ident_of("rt"),
                    self.ecx.ident_of("Piece")),
                vec!(self.ecx.lifetime(self.fmtsp,
                                       self.ecx.ident_of("static").name)),
                Vec::new()
            ), None);
        let ty = ast::TyFixedLengthVec(
            piece_ty,
            self.ecx.expr_uint(self.fmtsp, self.pieces.len())
        );
        let ty = self.ecx.ty(self.fmtsp, ty);
        let st = ast::ItemStatic(ty, ast::MutImmutable, fmt);
        let static_name = self.ecx.ident_of("__STATIC_FMTSTR");
        let item = self.ecx.item(self.fmtsp, static_name,
                                 self.static_attrs(), st);
        let decl = respan(self.fmtsp, ast::DeclItem(item));
        lets.push(@respan(self.fmtsp, ast::StmtDecl(@decl, ast::DUMMY_NODE_ID)));

        // Right now there is a bug such that for the expression:
        //      foo(bar(&1))
        // the lifetime of `1` doesn't outlast the call to `bar`, so it's not
        // vald for the call to `foo`. To work around this all arguments to the
        // format! string are shoved into locals. Furthermore, we shove the address
        // of each variable because we don't want to move out of the arguments
        // passed to this function.
        for (i, &e) in self.args.iter().enumerate() {
            if self.arg_types.get(i).is_none() {
                continue // error already generated
            }

            let name = self.ecx.ident_of(format!("__arg{}", i));
            pats.push(self.ecx.pat_ident(e.span, name));
            heads.push(self.ecx.expr_addr_of(e.span, e));
            locals.push(self.format_arg(e.span, Exact(i),
                                        self.ecx.expr_ident(e.span, name)));
        }
        for name in self.name_ordering.iter() {
            let e = match self.names.find(name) {
                Some(&e) if self.name_types.contains_key(name) => e,
                Some(..) | None => continue
            };

            let lname = self.ecx.ident_of(format!("__arg{}", *name));
            pats.push(self.ecx.pat_ident(e.span, lname));
            heads.push(self.ecx.expr_addr_of(e.span, e));
            *names.get_mut(*self.name_positions.get(name)) =
                Some(self.format_arg(e.span,
                                     Named((*name).clone()),
                                     self.ecx.expr_ident(e.span, lname)));
        }

        // Now create a vector containing all the arguments
        let slicename = self.ecx.ident_of("__args_vec");
        {
            let args = names.move_iter().map(|a| a.unwrap());
            let mut args = locals.move_iter().chain(args);
            let args = self.ecx.expr_vec_slice(self.fmtsp, args.collect());
            lets.push(self.ecx.stmt_let(self.fmtsp, false, slicename, args));
        }

        // Now create the fmt::Arguments struct with all our locals we created.
        let fmt = self.ecx.expr_ident(self.fmtsp, static_name);
        let args_slice = self.ecx.expr_ident(self.fmtsp, slicename);
        let result = self.ecx.expr_call_global(self.fmtsp, vec!(
                self.ecx.ident_of("std"),
                self.ecx.ident_of("fmt"),
                self.ecx.ident_of("Arguments"),
                self.ecx.ident_of("new")), vec!(fmt, args_slice));

        // We did all the work of making sure that the arguments
        // structure is safe, so we can safely have an unsafe block.
        let result = self.ecx.expr_block(P(ast::Block {
           view_items: Vec::new(),
           stmts: Vec::new(),
           expr: Some(result),
           id: ast::DUMMY_NODE_ID,
           rules: ast::UnsafeBlock(ast::CompilerGenerated),
           span: self.fmtsp,
        }));
        let resname = self.ecx.ident_of("__args");
        lets.push(self.ecx.stmt_let(self.fmtsp, false, resname, result));
        let res = self.ecx.expr_ident(self.fmtsp, resname);
        let result = self.ecx.expr_call(extra.span, extra, vec!(
                            self.ecx.expr_addr_of(extra.span, res)));
        let body = self.ecx.expr_block(self.ecx.block(self.fmtsp, lets,
                                                      Some(result)));

        // Constructs an AST equivalent to:
        //
        //      match (&arg0, &arg1) {
        //          (tmp0, tmp1) => body
        //      }
        //
        // It was:
        //
        //      let tmp0 = &arg0;
        //      let tmp1 = &arg1;
        //      body
        //
        // Because of #11585 the new temporary lifetime rule, the enclosing
        // statements for these temporaries become the let's themselves.
        // If one or more of them are RefCell's, RefCell borrow() will also
        // end there; they don't last long enough for body to use them. The
        // match expression solves the scope problem.
        //
        // Note, it may also very well be transformed to:
        //
        //      match arg0 {
        //          ref tmp0 => {
        //              match arg1 => {
        //                  ref tmp1 => body } } }
        //
        // But the nested match expression is proved to perform not as well
        // as series of let's; the first approach does.
        let pat = self.ecx.pat(self.fmtsp, ast::PatTup(pats));
        let arm = self.ecx.arm(self.fmtsp, vec!(pat), body);
        let head = self.ecx.expr(self.fmtsp, ast::ExprTup(heads));
        self.ecx.expr_match(self.fmtsp, head, vec!(arm))
    }

    fn format_arg(&self, sp: Span, argno: Position, arg: @ast::Expr)
                  -> @ast::Expr {
        let ty = match argno {
            Exact(ref i) => self.arg_types.get(*i).get_ref(),
            Named(ref s) => self.name_types.get(s)
        };

        let fmt_fn = match *ty {
            Known(ref tyname) => {
                match tyname.as_slice() {
                    ""  => "secret_show",
                    "?" => "secret_poly",
                    "b" => "secret_bool",
                    "c" => "secret_char",
                    "d" | "i" => "secret_signed",
                    "e" => "secret_lower_exp",
                    "E" => "secret_upper_exp",
                    "f" => "secret_float",
                    "o" => "secret_octal",
                    "p" => "secret_pointer",
                    "s" => "secret_string",
                    "t" => "secret_binary",
                    "u" => "secret_unsigned",
                    "x" => "secret_lower_hex",
                    "X" => "secret_upper_hex",
                    _ => {
                        self.ecx.span_err(sp, format!("unknown format trait `{}`",
                                                      *tyname));
                        "dummy"
                    }
                }
            }
            String => {
                return self.ecx.expr_call_global(sp, vec!(
                        self.ecx.ident_of("std"),
                        self.ecx.ident_of("fmt"),
                        self.ecx.ident_of("argumentstr")), vec!(arg))
            }
            Unsigned => {
                return self.ecx.expr_call_global(sp, vec!(
                        self.ecx.ident_of("std"),
                        self.ecx.ident_of("fmt"),
                        self.ecx.ident_of("argumentuint")), vec!(arg))
            }
        };

        let format_fn = self.ecx.path_global(sp, vec!(
                self.ecx.ident_of("std"),
                self.ecx.ident_of("fmt"),
                self.ecx.ident_of(fmt_fn)));
        self.ecx.expr_call_global(sp, vec!(
                self.ecx.ident_of("std"),
                self.ecx.ident_of("fmt"),
                self.ecx.ident_of("argument")), vec!(self.ecx.expr_path(format_fn), arg))
    }
}

pub fn expand_args(ecx: &mut ExtCtxt, sp: Span,
                   tts: &[ast::TokenTree]) -> Box<base::MacResult> {

    match parse_args(ecx, sp, tts) {
        (extra, Some((efmt, args, order, names))) => {
            MacExpr::new(expand_preparsed_format_args(ecx, sp, extra, efmt, args,
                                                order, names))
        }
        (_, None) => MacExpr::new(ecx.expr_uint(sp, 2))
    }
}

/// Take the various parts of `format_args!(extra, efmt, args...,
/// name=names...)` and construct the appropriate formatting
/// expression.
pub fn expand_preparsed_format_args(ecx: &mut ExtCtxt, sp: Span,
                                    extra: @ast::Expr,
                                    efmt: @ast::Expr, args: Vec<@ast::Expr>,
                                    name_ordering: Vec<~str>,
                                    names: HashMap<~str, @ast::Expr>) -> @ast::Expr {
    let arg_types = Vec::from_fn(args.len(), |_| None);
    let mut cx = Context {
        ecx: ecx,
        args: args,
        arg_types: arg_types,
        names: names,
        name_positions: HashMap::new(),
        name_types: HashMap::new(),
        name_ordering: name_ordering,
        nest_level: 0,
        next_arg: 0,
        pieces: Vec::new(),
        method_statics: Vec::new(),
        fmtsp: sp,
    };
    cx.fmtsp = efmt.span;
    let fmt = match expr_to_str(cx.ecx,
                                efmt,
                                "format argument must be a string literal.") {
        Some((fmt, _)) => fmt,
        None => return DummyResult::raw_expr(sp)
    };

    let mut parser = parse::Parser::new(fmt.get());
    loop {
        match parser.next() {
            Some(piece) => {
                if parser.errors.len() > 0 { break }
                cx.verify_piece(&piece);
                let piece = cx.trans_piece(&piece);
                cx.pieces.push(piece);
            }
            None => break
        }
    }
    match parser.errors.shift() {
        Some(error) => {
            cx.ecx.span_err(efmt.span, "invalid format string: " + error);
            return DummyResult::raw_expr(sp);
        }
        None => {}
    }

    // Make sure that all arguments were used and all arguments have types.
    for (i, ty) in cx.arg_types.iter().enumerate() {
        if ty.is_none() {
            cx.ecx.span_err(cx.args.get(i).span, "argument never used");
        }
    }
    for (name, e) in cx.names.iter() {
        if !cx.name_types.contains_key(name) {
            cx.ecx.span_err(e.span, "named argument never used");
        }
    }

    cx.to_expr(extra)
}
