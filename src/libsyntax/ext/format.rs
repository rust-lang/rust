// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use self::ArgumentType::*;
use self::Position::*;

use ast;
use codemap::{Span, respan};
use ext::base::*;
use ext::base;
use ext::build::AstBuilder;
use fmt_macros as parse;
use fold::Folder;
use parse::token::special_idents;
use parse::token;
use ptr::P;

use std::collections::HashMap;

#[derive(PartialEq)]
enum ArgumentType {
    Known(String),
    Unsigned
}

enum Position {
    Exact(usize),
    Named(String),
}

struct Context<'a, 'b:'a> {
    ecx: &'a mut ExtCtxt<'b>,
    /// The macro's call site. References to unstable formatting internals must
    /// use this span to pass the stability checker.
    macsp: Span,
    /// The span of the format string literal.
    fmtsp: Span,

    /// Parsed argument expressions and the types that we've found so far for
    /// them.
    args: Vec<P<ast::Expr>>,
    arg_types: Vec<Option<ArgumentType>>,
    /// Parsed named expressions and the types that we've found for them so far.
    /// Note that we keep a side-array of the ordering of the named arguments
    /// found to be sure that we can translate them in the same order that they
    /// were declared in.
    names: HashMap<String, P<ast::Expr>>,
    name_types: HashMap<String, ArgumentType>,
    name_ordering: Vec<String>,

    /// The latest consecutive literal strings, or empty if there weren't any.
    literal: String,

    /// Collection of the compiled `rt::Argument` structures
    pieces: Vec<P<ast::Expr>>,
    /// Collection of string literals
    str_pieces: Vec<P<ast::Expr>>,
    /// Stays `true` if all formatting parameters are default (as in "{}{}").
    all_pieces_simple: bool,

    name_positions: HashMap<String, usize>,

    /// Updated as arguments are consumed or methods are entered
    nest_level: usize,
    next_arg: usize,
}

/// Parses the arguments from the given list of tokens, returning None
/// if there's a parse error so we can continue parsing other format!
/// expressions.
///
/// If parsing succeeds, the return value is:
///
///     Some((fmtstr, unnamed arguments, ordering of named arguments,
///           named arguments))
fn parse_args(ecx: &mut ExtCtxt, sp: Span, tts: &[ast::TokenTree])
              -> Option<(P<ast::Expr>, Vec<P<ast::Expr>>, Vec<String>,
                         HashMap<String, P<ast::Expr>>)> {
    let mut args = Vec::new();
    let mut names = HashMap::<String, P<ast::Expr>>::new();
    let mut order = Vec::new();

    let mut p = ecx.new_parser_from_tts(tts);

    if p.token == token::Eof {
        ecx.span_err(sp, "requires at least a format string argument");
        return None;
    }
    let fmtstr = panictry!(p.parse_expr());
    let mut named = false;
    while p.token != token::Eof {
        if !panictry!(p.eat(&token::Comma)) {
            ecx.span_err(sp, "expected token: `,`");
            return None;
        }
        if p.token == token::Eof { break } // accept trailing commas
        if named || (p.token.is_ident() && p.look_ahead(1, |t| *t == token::Eq)) {
            named = true;
            let ident = match p.token {
                token::Ident(i, _) => {
                    panictry!(p.bump());
                    i
                }
                _ if named => {
                    ecx.span_err(p.span,
                                 "expected ident, positional arguments \
                                 cannot follow named arguments");
                    return None;
                }
                _ => {
                    ecx.span_err(p.span,
                                 &format!("expected ident for named argument, found `{}`",
                                         p.this_token_to_string()));
                    return None;
                }
            };
            let name: &str = &ident.name.as_str();

            panictry!(p.expect(&token::Eq));
            let e = panictry!(p.parse_expr());
            match names.get(name) {
                None => {}
                Some(prev) => {
                    ecx.span_err(e.span,
                                 &format!("duplicate argument named `{}`",
                                         name));
                    ecx.parse_sess.span_diagnostic.span_note(prev.span, "previously here");
                    continue
                }
            }
            order.push(name.to_string());
            names.insert(name.to_string(), e);
        } else {
            args.push(panictry!(p.parse_expr()));
        }
    }
    Some((fmtstr, args, order, names))
}

impl<'a, 'b> Context<'a, 'b> {
    /// Verifies one piece of a parse string. All errors are not emitted as
    /// fatal so we can continue giving errors about this and possibly other
    /// format strings.
    fn verify_piece(&mut self, p: &parse::Piece) {
        match *p {
            parse::String(..) => {}
            parse::NextArgument(ref arg) => {
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
                    parse::ArgumentNamed(s) => Named(s.to_string()),
                };

                let ty = Known(arg.format.ty.to_string());
                self.verify_arg_type(pos, ty);
            }
        }
    }

    fn verify_count(&mut self, c: parse::Count) {
        match c {
            parse::CountImplied | parse::CountIs(..) => {}
            parse::CountIsParam(i) => {
                self.verify_arg_type(Exact(i), Unsigned);
            }
            parse::CountIsName(s) => {
                self.verify_arg_type(Named(s.to_string()), Unsigned);
            }
            parse::CountIsNextParam => {
                if self.check_positional_ok() {
                    let next_arg = self.next_arg;
                    self.verify_arg_type(Exact(next_arg), Unsigned);
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

    fn describe_num_args(&self) -> String {
        match self.args.len() {
            0 => "no arguments given".to_string(),
            1 => "there is 1 argument".to_string(),
            x => format!("there are {} arguments", x),
        }
    }

    fn verify_arg_type(&mut self, arg: Position, ty: ArgumentType) {
        match arg {
            Exact(arg) => {
                if self.args.len() <= arg {
                    let msg = format!("invalid reference to argument `{}` ({})",
                                      arg, self.describe_num_args());

                    self.ecx.span_err(self.fmtsp, &msg[..]);
                    return;
                }
                {
                    let arg_type = match self.arg_types[arg] {
                        None => None,
                        Some(ref x) => Some(x)
                    };
                    self.verify_same(self.args[arg].span, &ty, arg_type);
                }
                if self.arg_types[arg].is_none() {
                    self.arg_types[arg] = Some(ty);
                }
            }

            Named(name) => {
                let span = match self.names.get(&name) {
                    Some(e) => e.span,
                    None => {
                        let msg = format!("there is no argument named `{}`", name);
                        self.ecx.span_err(self.fmtsp, &msg[..]);
                        return;
                    }
                };
                self.verify_same(span, &ty, self.name_types.get(&name));
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
                                  &format!("argument redeclared with type `{}` when \
                                           it was previously `{}`",
                                          *ty,
                                          *cur));
            }
            (&Known(ref cur), _) => {
                self.ecx.span_err(sp,
                                  &format!("argument used to format with `{}` was \
                                           attempted to not be used for formatting",
                                           *cur));
            }
            (_, &Known(ref ty)) => {
                self.ecx.span_err(sp,
                                  &format!("argument previously used as a format \
                                           argument attempted to be used as `{}`",
                                           *ty));
            }
            (_, _) => {
                self.ecx.span_err(sp, "argument declared with multiple formats");
            }
        }
    }

    fn rtpath(ecx: &ExtCtxt, s: &str) -> Vec<ast::Ident> {
        ecx.std_path(&["fmt", "rt", "v1", s])
    }

    fn trans_count(&self, c: parse::Count) -> P<ast::Expr> {
        let sp = self.macsp;
        let count = |c, arg| {
            let mut path = Context::rtpath(self.ecx, "Count");
            path.push(self.ecx.ident_of(c));
            match arg {
                Some(arg) => self.ecx.expr_call_global(sp, path, vec![arg]),
                None => self.ecx.expr_path(self.ecx.path_global(sp, path)),
            }
        };
        match c {
            parse::CountIs(i) => count("Is", Some(self.ecx.expr_usize(sp, i))),
            parse::CountIsParam(i) => {
                count("Param", Some(self.ecx.expr_usize(sp, i)))
            }
            parse::CountImplied => count("Implied", None),
            parse::CountIsNextParam => count("NextParam", None),
            parse::CountIsName(n) => {
                let i = match self.name_positions.get(n) {
                    Some(&i) => i,
                    None => 0, // error already emitted elsewhere
                };
                let i = i + self.args.len();
                count("Param", Some(self.ecx.expr_usize(sp, i)))
            }
        }
    }

    /// Translate the accumulated string literals to a literal expression
    fn trans_literal_string(&mut self) -> P<ast::Expr> {
        let sp = self.fmtsp;
        let s = token::intern_and_get_ident(&self.literal);
        self.literal.clear();
        self.ecx.expr_str(sp, s)
    }

    /// Translate a `parse::Piece` to a static `rt::Argument` or append
    /// to the `literal` string.
    fn trans_piece(&mut self, piece: &parse::Piece) -> Option<P<ast::Expr>> {
        let sp = self.macsp;
        match *piece {
            parse::String(s) => {
                self.literal.push_str(s);
                None
            }
            parse::NextArgument(ref arg) => {
                // Translate the position
                let pos = {
                    let pos = |c, arg| {
                        let mut path = Context::rtpath(self.ecx, "Position");
                        path.push(self.ecx.ident_of(c));
                        match arg {
                            Some(i) => {
                                let arg = self.ecx.expr_usize(sp, i);
                                self.ecx.expr_call_global(sp, path, vec![arg])
                            }
                            None => {
                                self.ecx.expr_path(self.ecx.path_global(sp, path))
                            }
                        }
                    };
                    match arg.position {
                        // These two have a direct mapping
                        parse::ArgumentNext => pos("Next", None),
                        parse::ArgumentIs(i) => pos("At", Some(i)),

                        // Named arguments are converted to positional arguments
                        // at the end of the list of arguments
                        parse::ArgumentNamed(n) => {
                            let i = match self.name_positions.get(n) {
                                Some(&i) => i,
                                None => 0, // error already emitted elsewhere
                            };
                            let i = i + self.args.len();
                            pos("At", Some(i))
                        }
                    }
                };

                let simple_arg = parse::Argument {
                    position: parse::ArgumentNext,
                    format: parse::FormatSpec {
                        fill: arg.format.fill,
                        align: parse::AlignUnknown,
                        flags: 0,
                        precision: parse::CountImplied,
                        width: parse::CountImplied,
                        ty: arg.format.ty
                    }
                };

                let fill = match arg.format.fill { Some(c) => c, None => ' ' };

                if *arg != simple_arg || fill != ' ' {
                    self.all_pieces_simple = false;
                }

                // Translate the format
                let fill = self.ecx.expr_lit(sp, ast::LitChar(fill));
                let align = |name| {
                    let mut p = Context::rtpath(self.ecx, "Alignment");
                    p.push(self.ecx.ident_of(name));
                    self.ecx.path_global(sp, p)
                };
                let align = match arg.format.align {
                    parse::AlignLeft => align("Left"),
                    parse::AlignRight => align("Right"),
                    parse::AlignCenter => align("Center"),
                    parse::AlignUnknown => align("Unknown"),
                };
                let align = self.ecx.expr_path(align);
                let flags = self.ecx.expr_u32(sp, arg.format.flags);
                let prec = self.trans_count(arg.format.precision);
                let width = self.trans_count(arg.format.width);
                let path = self.ecx.path_global(sp, Context::rtpath(self.ecx, "FormatSpec"));
                let fmt = self.ecx.expr_struct(sp, path, vec!(
                    self.ecx.field_imm(sp, self.ecx.ident_of("fill"), fill),
                    self.ecx.field_imm(sp, self.ecx.ident_of("align"), align),
                    self.ecx.field_imm(sp, self.ecx.ident_of("flags"), flags),
                    self.ecx.field_imm(sp, self.ecx.ident_of("precision"), prec),
                    self.ecx.field_imm(sp, self.ecx.ident_of("width"), width)));

                let path = self.ecx.path_global(sp, Context::rtpath(self.ecx, "Argument"));
                Some(self.ecx.expr_struct(sp, path, vec!(
                    self.ecx.field_imm(sp, self.ecx.ident_of("position"), pos),
                    self.ecx.field_imm(sp, self.ecx.ident_of("format"), fmt))))
            }
        }
    }

    fn static_array(ecx: &mut ExtCtxt,
                    name: &str,
                    piece_ty: P<ast::Ty>,
                    pieces: Vec<P<ast::Expr>>)
                    -> P<ast::Expr> {
        let sp = piece_ty.span;
        let ty = ecx.ty_rptr(sp,
            ecx.ty(sp, ast::TyVec(piece_ty)),
            Some(ecx.lifetime(sp, special_idents::static_lifetime.name)),
            ast::MutImmutable);
        let slice = ecx.expr_vec_slice(sp, pieces);
        // static instead of const to speed up codegen by not requiring this to be inlined
        let st = ast::ItemStatic(ty, ast::MutImmutable, slice);

        let name = ecx.ident_of(name);
        let item = ecx.item(sp, name, vec![], st);
        let decl = respan(sp, ast::DeclItem(item));

        // Wrap the declaration in a block so that it forms a single expression.
        ecx.expr_block(ecx.block(sp,
            vec![P(respan(sp, ast::StmtDecl(P(decl), ast::DUMMY_NODE_ID)))],
            Some(ecx.expr_ident(sp, name))))
    }

    /// Actually builds the expression which the iformat! block will be expanded
    /// to
    fn into_expr(mut self) -> P<ast::Expr> {
        let mut locals = Vec::new();
        let mut names = vec![None; self.name_positions.len()];
        let mut pats = Vec::new();
        let mut heads = Vec::new();

        // First, build up the static array which will become our precompiled
        // format "string"
        let static_lifetime = self.ecx.lifetime(self.fmtsp, special_idents::static_lifetime.name);
        let piece_ty = self.ecx.ty_rptr(
                self.fmtsp,
                self.ecx.ty_ident(self.fmtsp, self.ecx.ident_of("str")),
                Some(static_lifetime),
                ast::MutImmutable);
        let pieces = Context::static_array(self.ecx,
                                           "__STATIC_FMTSTR",
                                           piece_ty,
                                           self.str_pieces);


        // Right now there is a bug such that for the expression:
        //      foo(bar(&1))
        // the lifetime of `1` doesn't outlast the call to `bar`, so it's not
        // valid for the call to `foo`. To work around this all arguments to the
        // format! string are shoved into locals. Furthermore, we shove the address
        // of each variable because we don't want to move out of the arguments
        // passed to this function.
        for (i, e) in self.args.into_iter().enumerate() {
            let arg_ty = match self.arg_types[i].as_ref() {
                Some(ty) => ty,
                None => continue // error already generated
            };

            let name = self.ecx.ident_of(&format!("__arg{}", i));
            pats.push(self.ecx.pat_ident(e.span, name));
            locals.push(Context::format_arg(self.ecx, self.macsp, e.span, arg_ty,
                                            self.ecx.expr_ident(e.span, name)));
            heads.push(self.ecx.expr_addr_of(e.span, e));
        }
        for name in &self.name_ordering {
            let e = match self.names.remove(name) {
                Some(e) => e,
                None => continue
            };
            let arg_ty = match self.name_types.get(name) {
                Some(ty) => ty,
                None => continue
            };

            let lname = self.ecx.ident_of(&format!("__arg{}",
                                                  *name));
            pats.push(self.ecx.pat_ident(e.span, lname));
            names[*self.name_positions.get(name).unwrap()] =
                Some(Context::format_arg(self.ecx, self.macsp, e.span, arg_ty,
                                         self.ecx.expr_ident(e.span, lname)));
            heads.push(self.ecx.expr_addr_of(e.span, e));
        }

        // Now create a vector containing all the arguments
        let args = locals.into_iter().chain(names.into_iter().map(|a| a.unwrap()));

        let args_array = self.ecx.expr_vec(self.fmtsp, args.collect());

        // Constructs an AST equivalent to:
        //
        //      match (&arg0, &arg1) {
        //          (tmp0, tmp1) => args_array
        //      }
        //
        // It was:
        //
        //      let tmp0 = &arg0;
        //      let tmp1 = &arg1;
        //      args_array
        //
        // Because of #11585 the new temporary lifetime rule, the enclosing
        // statements for these temporaries become the let's themselves.
        // If one or more of them are RefCell's, RefCell borrow() will also
        // end there; they don't last long enough for args_array to use them.
        // The match expression solves the scope problem.
        //
        // Note, it may also very well be transformed to:
        //
        //      match arg0 {
        //          ref tmp0 => {
        //              match arg1 => {
        //                  ref tmp1 => args_array } } }
        //
        // But the nested match expression is proved to perform not as well
        // as series of let's; the first approach does.
        let pat = self.ecx.pat_tuple(self.fmtsp, pats);
        let arm = self.ecx.arm(self.fmtsp, vec!(pat), args_array);
        let head = self.ecx.expr(self.fmtsp, ast::ExprTup(heads));
        let result = self.ecx.expr_match(self.fmtsp, head, vec!(arm));

        let args_slice = self.ecx.expr_addr_of(self.fmtsp, result);

        // Now create the fmt::Arguments struct with all our locals we created.
        let (fn_name, fn_args) = if self.all_pieces_simple {
            ("new_v1", vec![pieces, args_slice])
        } else {
            // Build up the static array which will store our precompiled
            // nonstandard placeholders, if there are any.
            let piece_ty = self.ecx.ty_path(self.ecx.path_global(
                    self.macsp,
                    Context::rtpath(self.ecx, "Argument")));
            let fmt = Context::static_array(self.ecx,
                                            "__STATIC_FMTARGS",
                                            piece_ty,
                                            self.pieces);

            ("new_v1_formatted", vec![pieces, args_slice, fmt])
        };

        let path = self.ecx.std_path(&["fmt", "Arguments", fn_name]);
        self.ecx.expr_call_global(self.macsp, path, fn_args)
    }

    fn format_arg(ecx: &ExtCtxt, macsp: Span, sp: Span,
                  ty: &ArgumentType, arg: P<ast::Expr>)
                  -> P<ast::Expr> {
        let trait_ = match *ty {
            Known(ref tyname) => {
                match &tyname[..] {
                    ""  => "Display",
                    "?" => "Debug",
                    "e" => "LowerExp",
                    "E" => "UpperExp",
                    "o" => "Octal",
                    "p" => "Pointer",
                    "b" => "Binary",
                    "x" => "LowerHex",
                    "X" => "UpperHex",
                    _ => {
                        ecx.span_err(sp,
                                     &format!("unknown format trait `{}`",
                                             *tyname));
                        "Dummy"
                    }
                }
            }
            Unsigned => {
                let path = ecx.std_path(&["fmt", "ArgumentV1", "from_usize"]);
                return ecx.expr_call_global(macsp, path, vec![arg])
            }
        };

        let path = ecx.std_path(&["fmt", trait_, "fmt"]);
        let format_fn = ecx.path_global(sp, path);
        let path = ecx.std_path(&["fmt", "ArgumentV1", "new"]);
        ecx.expr_call_global(macsp, path, vec![arg, ecx.expr_path(format_fn)])
    }
}

pub fn expand_format_args<'cx>(ecx: &'cx mut ExtCtxt, sp: Span,
                               tts: &[ast::TokenTree])
                               -> Box<base::MacResult+'cx> {

    match parse_args(ecx, sp, tts) {
        Some((efmt, args, order, names)) => {
            MacEager::expr(expand_preparsed_format_args(ecx, sp, efmt,
                                                      args, order, names))
        }
        None => DummyResult::expr(sp)
    }
}

/// Take the various parts of `format_args!(efmt, args..., name=names...)`
/// and construct the appropriate formatting expression.
pub fn expand_preparsed_format_args(ecx: &mut ExtCtxt, sp: Span,
                                    efmt: P<ast::Expr>,
                                    args: Vec<P<ast::Expr>>,
                                    name_ordering: Vec<String>,
                                    names: HashMap<String, P<ast::Expr>>)
                                    -> P<ast::Expr> {
    let arg_types: Vec<_> = (0..args.len()).map(|_| None).collect();
    let macsp = ecx.call_site();
    // Expand the format literal so that efmt.span will have a backtrace. This
    // is essential for locating a bug when the format literal is generated in
    // a macro. (e.g. println!("{}"), which uses concat!($fmt, "\n")).
    let efmt = ecx.expander().fold_expr(efmt);
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
        literal: String::new(),
        pieces: Vec::new(),
        str_pieces: Vec::new(),
        all_pieces_simple: true,
        macsp: macsp,
        fmtsp: efmt.span,
    };
    let fmt = match expr_to_string(cx.ecx,
                                   efmt,
                                   "format argument must be a string literal.") {
        Some((fmt, _)) => fmt,
        None => return DummyResult::raw_expr(sp)
    };

    let mut parser = parse::Parser::new(&fmt);

    loop {
        match parser.next() {
            Some(piece) => {
                if !parser.errors.is_empty() { break }
                cx.verify_piece(&piece);
                match cx.trans_piece(&piece) {
                    Some(piece) => {
                        let s = cx.trans_literal_string();
                        cx.str_pieces.push(s);
                        cx.pieces.push(piece);
                    }
                    None => {}
                }
            }
            None => break
        }
    }
    if !parser.errors.is_empty() {
        cx.ecx.span_err(cx.fmtsp, &format!("invalid format string: {}",
                                          parser.errors.remove(0)));
        return DummyResult::raw_expr(sp);
    }
    if !cx.literal.is_empty() {
        let s = cx.trans_literal_string();
        cx.str_pieces.push(s);
    }

    // Make sure that all arguments were used and all arguments have types.
    for (i, ty) in cx.arg_types.iter().enumerate() {
        if ty.is_none() {
            cx.ecx.span_err(cx.args[i].span, "argument never used");
        }
    }
    for (name, e) in &cx.names {
        if !cx.name_types.contains_key(name) {
            cx.ecx.span_err(e.span, "named argument never used");
        }
    }

    cx.into_expr()
}
