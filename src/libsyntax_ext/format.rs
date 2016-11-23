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

use fmt_macros as parse;

use syntax::ast;
use syntax::ext::base::*;
use syntax::ext::base;
use syntax::ext::build::AstBuilder;
use syntax::parse::token;
use syntax::ptr::P;
use syntax::symbol::{Symbol, keywords};
use syntax_pos::{Span, DUMMY_SP};
use syntax::tokenstream;

use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Entry;

#[derive(PartialEq)]
enum ArgumentType {
    Placeholder(String),
    Count,
}

enum Position {
    Exact(usize),
    Named(String),
}

struct Context<'a, 'b: 'a> {
    ecx: &'a mut ExtCtxt<'b>,
    /// The macro's call site. References to unstable formatting internals must
    /// use this span to pass the stability checker.
    macsp: Span,
    /// The span of the format string literal.
    fmtsp: Span,

    /// List of parsed argument expressions.
    /// Named expressions are resolved early, and are appended to the end of
    /// argument expressions.
    ///
    /// Example showing the various data structures in motion:
    ///
    /// * Original: `"{foo:o} {:o} {foo:x} {0:x} {1:o} {:x} {1:x} {0:o}"`
    /// * Implicit argument resolution: `"{foo:o} {0:o} {foo:x} {0:x} {1:o} {1:x} {1:x} {0:o}"`
    /// * Name resolution: `"{2:o} {0:o} {2:x} {0:x} {1:o} {1:x} {1:x} {0:o}"`
    /// * `arg_types` (in JSON): `[[0, 1, 0], [0, 1, 1], [0, 1]]`
    /// * `arg_unique_types` (in simplified JSON): `[["o", "x"], ["o", "x"], ["o", "x"]]`
    /// * `names` (in JSON): `{"foo": 2}`
    args: Vec<P<ast::Expr>>,
    /// Placeholder slot numbers indexed by argument.
    arg_types: Vec<Vec<usize>>,
    /// Unique format specs seen for each argument.
    arg_unique_types: Vec<Vec<ArgumentType>>,
    /// Map from named arguments to their resolved indices.
    names: HashMap<String, usize>,

    /// The latest consecutive literal strings, or empty if there weren't any.
    literal: String,

    /// Collection of the compiled `rt::Argument` structures
    pieces: Vec<P<ast::Expr>>,
    /// Collection of string literals
    str_pieces: Vec<P<ast::Expr>>,
    /// Stays `true` if all formatting parameters are default (as in "{}{}").
    all_pieces_simple: bool,

    /// Mapping between positional argument references and indices into the
    /// final generated static argument array. We record the starting indices
    /// corresponding to each positional argument, and number of references
    /// consumed so far for each argument, to facilitate correct `Position`
    /// mapping in `trans_piece`. In effect this can be seen as a "flattened"
    /// version of `arg_unique_types`.
    ///
    /// Again with the example described above in docstring for `args`:
    ///
    /// * `arg_index_map` (in JSON): `[[0, 1, 0], [2, 3, 3], [4, 5]]`
    arg_index_map: Vec<Vec<usize>>,

    /// Starting offset of count argument slots.
    count_args_index_offset: usize,

    /// Count argument slots and tracking data structures.
    /// Count arguments are separately tracked for de-duplication in case
    /// multiple references are made to one argument. For example, in this
    /// format string:
    ///
    /// * Original: `"{:.*} {:.foo$} {1:.*} {:.0$}"`
    /// * Implicit argument resolution: `"{1:.0$} {2:.foo$} {1:.3$} {4:.0$}"`
    /// * Name resolution: `"{1:.0$} {2:.5$} {1:.3$} {4:.0$}"`
    /// * `count_positions` (in JSON): `{0: 0, 5: 1, 3: 2}`
    /// * `count_args`: `vec![Exact(0), Exact(5), Exact(3)]`
    count_args: Vec<Position>,
    /// Relative slot numbers for count arguments.
    count_positions: HashMap<usize, usize>,
    /// Number of count slots assigned.
    count_positions_count: usize,

    /// Current position of the implicit positional arg pointer, as if it
    /// still existed in this phase of processing.
    /// Used only for `all_pieces_simple` tracking in `trans_piece`.
    curarg: usize,
}

/// Parses the arguments from the given list of tokens, returning None
/// if there's a parse error so we can continue parsing other format!
/// expressions.
///
/// If parsing succeeds, the return value is:
/// ```ignore
/// Some((fmtstr, parsed arguments, index map for named arguments))
/// ```
fn parse_args(ecx: &mut ExtCtxt,
              sp: Span,
              tts: &[tokenstream::TokenTree])
              -> Option<(P<ast::Expr>, Vec<P<ast::Expr>>, HashMap<String, usize>)> {
    let mut args = Vec::<P<ast::Expr>>::new();
    let mut names = HashMap::<String, usize>::new();

    let mut p = ecx.new_parser_from_tts(tts);

    if p.token == token::Eof {
        ecx.span_err(sp, "requires at least a format string argument");
        return None;
    }
    let fmtstr = panictry!(p.parse_expr());
    let mut named = false;
    while p.token != token::Eof {
        if !p.eat(&token::Comma) {
            ecx.span_err(sp, "expected token: `,`");
            return None;
        }
        if p.token == token::Eof {
            break;
        } // accept trailing commas
        if named || (p.token.is_ident() && p.look_ahead(1, |t| *t == token::Eq)) {
            named = true;
            let ident = match p.token {
                token::Ident(i) => {
                    p.bump();
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
            if let Some(prev) = names.get(name) {
                ecx.struct_span_err(e.span, &format!("duplicate argument named `{}`", name))
                    .span_note(args[*prev].span, "previously here")
                    .emit();
                continue;
            }

            // Resolve names into slots early.
            // Since all the positional args are already seen at this point
            // if the input is valid, we can simply append to the positional
            // args. And remember the names.
            let slot = args.len();
            names.insert(name.to_string(), slot);
            args.push(e);
        } else {
            args.push(panictry!(p.parse_expr()));
        }
    }
    Some((fmtstr, args, names))
}

impl<'a, 'b> Context<'a, 'b> {
    fn resolve_name_inplace(&self, p: &mut parse::Piece) {
        // NOTE: the `unwrap_or` branch is needed in case of invalid format
        // arguments, e.g. `format_args!("{foo}")`.
        let lookup = |s| *self.names.get(s).unwrap_or(&0);

        match *p {
            parse::String(_) => {}
            parse::NextArgument(ref mut arg) => {
                if let parse::ArgumentNamed(s) = arg.position {
                    arg.position = parse::ArgumentIs(lookup(s));
                }
                if let parse::CountIsName(s) = arg.format.width {
                    arg.format.width = parse::CountIsParam(lookup(s));
                }
                if let parse::CountIsName(s) = arg.format.precision {
                    arg.format.precision = parse::CountIsParam(lookup(s));
                }
            }
        }
    }

    /// Verifies one piece of a parse string, and remembers it if valid.
    /// All errors are not emitted as fatal so we can continue giving errors
    /// about this and possibly other format strings.
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
                    parse::ArgumentIs(i) => Exact(i),
                    parse::ArgumentNamed(s) => Named(s.to_string()),
                };

                let ty = Placeholder(arg.format.ty.to_string());
                self.verify_arg_type(pos, ty);
            }
        }
    }

    fn verify_count(&mut self, c: parse::Count) {
        match c {
            parse::CountImplied |
            parse::CountIs(..) => {}
            parse::CountIsParam(i) => {
                self.verify_arg_type(Exact(i), Count);
            }
            parse::CountIsName(s) => {
                self.verify_arg_type(Named(s.to_string()), Count);
            }
        }
    }

    fn describe_num_args(&self) -> String {
        match self.args.len() {
            0 => "no arguments given".to_string(),
            1 => "there is 1 argument".to_string(),
            x => format!("there are {} arguments", x),
        }
    }

    /// Actually verifies and tracks a given format placeholder
    /// (a.k.a. argument).
    fn verify_arg_type(&mut self, arg: Position, ty: ArgumentType) {
        match arg {
            Exact(arg) => {
                if self.args.len() <= arg {
                    let msg = format!("invalid reference to argument `{}` ({})",
                                      arg,
                                      self.describe_num_args());

                    self.ecx.span_err(self.fmtsp, &msg[..]);
                    return;
                }
                match ty {
                    Placeholder(_) => {
                        // record every (position, type) combination only once
                        let ref mut seen_ty = self.arg_unique_types[arg];
                        let i = match seen_ty.iter().position(|x| *x == ty) {
                            Some(i) => i,
                            None => {
                                let i = seen_ty.len();
                                seen_ty.push(ty);
                                i
                            }
                        };
                        self.arg_types[arg].push(i);
                    }
                    Count => {
                        match self.count_positions.entry(arg) {
                            Entry::Vacant(e) => {
                                let i = self.count_positions_count;
                                e.insert(i);
                                self.count_args.push(Exact(arg));
                                self.count_positions_count += 1;
                            }
                            Entry::Occupied(_) => {}
                        }
                    }
                }
            }

            Named(name) => {
                let idx = match self.names.get(&name) {
                    Some(e) => *e,
                    None => {
                        let msg = format!("there is no argument named `{}`", name);
                        self.ecx.span_err(self.fmtsp, &msg[..]);
                        return;
                    }
                };
                // Treat as positional arg.
                self.verify_arg_type(Exact(idx), ty)
            }
        }
    }

    /// Builds the mapping between format placeholders and argument objects.
    fn build_index_map(&mut self) {
        // NOTE: Keep the ordering the same as `into_expr`'s expansion would do!
        let args_len = self.args.len();
        self.arg_index_map.reserve(args_len);

        let mut sofar = 0usize;

        // Map the arguments
        for i in 0..args_len {
            let ref arg_types = self.arg_types[i];
            let mut arg_offsets = Vec::with_capacity(arg_types.len());
            for offset in arg_types {
                arg_offsets.push(sofar + *offset);
            }
            self.arg_index_map.push(arg_offsets);
            sofar += self.arg_unique_types[i].len();
        }

        // Record starting index for counts, which appear just after arguments
        self.count_args_index_offset = sofar;
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
                // This needs mapping too, as `i` is referring to a macro
                // argument.
                let i = match self.count_positions.get(&i) {
                    Some(&i) => i,
                    None => 0, // error already emitted elsewhere
                };
                let i = i + self.count_args_index_offset;
                count("Param", Some(self.ecx.expr_usize(sp, i)))
            }
            parse::CountImplied => count("Implied", None),
            // should never be the case, names are already resolved
            parse::CountIsName(_) => panic!("should never happen"),
        }
    }

    /// Translate the accumulated string literals to a literal expression
    fn trans_literal_string(&mut self) -> P<ast::Expr> {
        let sp = self.fmtsp;
        let s = Symbol::intern(&self.literal);
        self.literal.clear();
        self.ecx.expr_str(sp, s)
    }

    /// Translate a `parse::Piece` to a static `rt::Argument` or append
    /// to the `literal` string.
    fn trans_piece(&mut self,
                   piece: &parse::Piece,
                   arg_index_consumed: &mut Vec<usize>)
                   -> Option<P<ast::Expr>> {
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
                            None => self.ecx.expr_path(self.ecx.path_global(sp, path)),
                        }
                    };
                    match arg.position {
                        parse::ArgumentIs(i) => {
                            // Map to index in final generated argument array
                            // in case of multiple types specified
                            let arg_idx = match arg_index_consumed.get_mut(i) {
                                None => 0, // error already emitted elsewhere
                                Some(offset) => {
                                    let ref idx_map = self.arg_index_map[i];
                                    // unwrap_or branch: error already emitted elsewhere
                                    let arg_idx = *idx_map.get(*offset).unwrap_or(&0);
                                    *offset += 1;
                                    arg_idx
                                }
                            };
                            pos("At", Some(arg_idx))
                        }

                        // should never be the case, because names are already
                        // resolved.
                        parse::ArgumentNamed(_) => panic!("should never happen"),
                    }
                };

                let simple_arg = parse::Argument {
                    position: {
                        // We don't have ArgumentNext any more, so we have to
                        // track the current argument ourselves.
                        let i = self.curarg;
                        self.curarg += 1;
                        parse::ArgumentIs(i)
                    },
                    format: parse::FormatSpec {
                        fill: arg.format.fill,
                        align: parse::AlignUnknown,
                        flags: 0,
                        precision: parse::CountImplied,
                        width: parse::CountImplied,
                        ty: arg.format.ty,
                    },
                };

                let fill = match arg.format.fill {
                    Some(c) => c,
                    None => ' ',
                };

                if *arg != simple_arg || fill != ' ' {
                    self.all_pieces_simple = false;
                }

                // Translate the format
                let fill = self.ecx.expr_lit(sp, ast::LitKind::Char(fill));
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
                let fmt =
                    self.ecx.expr_struct(sp,
                                         path,
                                         vec![self.ecx
                                                  .field_imm(sp, self.ecx.ident_of("fill"), fill),
                                              self.ecx.field_imm(sp,
                                                                 self.ecx.ident_of("align"),
                                                                 align),
                                              self.ecx.field_imm(sp,
                                                                 self.ecx.ident_of("flags"),
                                                                 flags),
                                              self.ecx.field_imm(sp,
                                                                 self.ecx.ident_of("precision"),
                                                                 prec),
                                              self.ecx.field_imm(sp,
                                                                 self.ecx.ident_of("width"),
                                                                 width)]);

                let path = self.ecx.path_global(sp, Context::rtpath(self.ecx, "Argument"));
                Some(self.ecx.expr_struct(sp,
                                          path,
                                          vec![self.ecx.field_imm(sp,
                                                                  self.ecx.ident_of("position"),
                                                                  pos),
                                               self.ecx.field_imm(sp,
                                                                  self.ecx.ident_of("format"),
                                                                  fmt)]))
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
                             ecx.ty(sp, ast::TyKind::Slice(piece_ty)),
                             Some(ecx.lifetime(sp, keywords::StaticLifetime.name())),
                             ast::Mutability::Immutable);
        let slice = ecx.expr_vec_slice(sp, pieces);
        // static instead of const to speed up codegen by not requiring this to be inlined
        let st = ast::ItemKind::Static(ty, ast::Mutability::Immutable, slice);

        let name = ecx.ident_of(name);
        let item = ecx.item(sp, name, vec![], st);
        let stmt = ast::Stmt {
            id: ast::DUMMY_NODE_ID,
            node: ast::StmtKind::Item(item),
            span: sp,
        };

        // Wrap the declaration in a block so that it forms a single expression.
        ecx.expr_block(ecx.block(sp, vec![stmt, ecx.stmt_expr(ecx.expr_ident(sp, name))]))
    }

    /// Actually builds the expression which the format_args! block will be
    /// expanded to
    fn into_expr(mut self) -> P<ast::Expr> {
        let mut locals = Vec::new();
        let mut counts = Vec::new();
        let mut pats = Vec::new();
        let mut heads = Vec::new();

        // First, build up the static array which will become our precompiled
        // format "string"
        let static_lifetime = self.ecx.lifetime(self.fmtsp, keywords::StaticLifetime.name());
        let piece_ty = self.ecx.ty_rptr(self.fmtsp,
                                        self.ecx.ty_ident(self.fmtsp, self.ecx.ident_of("str")),
                                        Some(static_lifetime),
                                        ast::Mutability::Immutable);
        let pieces = Context::static_array(self.ecx, "__STATIC_FMTSTR", piece_ty, self.str_pieces);

        // Before consuming the expressions, we have to remember spans for
        // count arguments as they are now generated separate from other
        // arguments, hence have no access to the `P<ast::Expr>`'s.
        let spans_pos: Vec<_> = self.args.iter().map(|e| e.span.clone()).collect();

        // Right now there is a bug such that for the expression:
        //      foo(bar(&1))
        // the lifetime of `1` doesn't outlast the call to `bar`, so it's not
        // valid for the call to `foo`. To work around this all arguments to the
        // format! string are shoved into locals. Furthermore, we shove the address
        // of each variable because we don't want to move out of the arguments
        // passed to this function.
        for (i, e) in self.args.into_iter().enumerate() {
            let name = self.ecx.ident_of(&format!("__arg{}", i));
            pats.push(self.ecx.pat_ident(DUMMY_SP, name));
            for ref arg_ty in self.arg_unique_types[i].iter() {
                locals.push(Context::format_arg(self.ecx,
                                                self.macsp,
                                                e.span,
                                                arg_ty,
                                                self.ecx.expr_ident(e.span, name)));
            }
            heads.push(self.ecx.expr_addr_of(e.span, e));
        }
        for pos in self.count_args {
            let name = self.ecx.ident_of(&match pos {
                Exact(i) => format!("__arg{}", i),
                _ => panic!("should never happen"),
            });
            let span = match pos {
                Exact(i) => spans_pos[i],
                _ => panic!("should never happen"),
            };
            counts.push(Context::format_arg(self.ecx,
                                            self.macsp,
                                            span,
                                            &Count,
                                            self.ecx.expr_ident(span, name)));
        }

        // Now create a vector containing all the arguments
        let args = locals.into_iter().chain(counts.into_iter());

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
        let arm = self.ecx.arm(self.fmtsp, vec![pat], args_array);
        let head = self.ecx.expr(self.fmtsp, ast::ExprKind::Tup(heads));
        let result = self.ecx.expr_match(self.fmtsp, head, vec![arm]);

        let args_slice = self.ecx.expr_addr_of(self.fmtsp, result);

        // Now create the fmt::Arguments struct with all our locals we created.
        let (fn_name, fn_args) = if self.all_pieces_simple {
            ("new_v1", vec![pieces, args_slice])
        } else {
            // Build up the static array which will store our precompiled
            // nonstandard placeholders, if there are any.
            let piece_ty = self.ecx
                .ty_path(self.ecx.path_global(self.macsp, Context::rtpath(self.ecx, "Argument")));
            let fmt = Context::static_array(self.ecx, "__STATIC_FMTARGS", piece_ty, self.pieces);

            ("new_v1_formatted", vec![pieces, args_slice, fmt])
        };

        let path = self.ecx.std_path(&["fmt", "Arguments", fn_name]);
        self.ecx.expr_call_global(self.macsp, path, fn_args)
    }

    fn format_arg(ecx: &ExtCtxt,
                  macsp: Span,
                  sp: Span,
                  ty: &ArgumentType,
                  arg: P<ast::Expr>)
                  -> P<ast::Expr> {
        let trait_ = match *ty {
            Placeholder(ref tyname) => {
                match &tyname[..] {
                    "" => "Display",
                    "?" => "Debug",
                    "e" => "LowerExp",
                    "E" => "UpperExp",
                    "o" => "Octal",
                    "p" => "Pointer",
                    "b" => "Binary",
                    "x" => "LowerHex",
                    "X" => "UpperHex",
                    _ => {
                        ecx.span_err(sp, &format!("unknown format trait `{}`", *tyname));
                        "Dummy"
                    }
                }
            }
            Count => {
                let path = ecx.std_path(&["fmt", "ArgumentV1", "from_usize"]);
                return ecx.expr_call_global(macsp, path, vec![arg]);
            }
        };

        let path = ecx.std_path(&["fmt", trait_, "fmt"]);
        let format_fn = ecx.path_global(sp, path);
        let path = ecx.std_path(&["fmt", "ArgumentV1", "new"]);
        ecx.expr_call_global(macsp, path, vec![arg, ecx.expr_path(format_fn)])
    }
}

pub fn expand_format_args<'cx>(ecx: &'cx mut ExtCtxt,
                               sp: Span,
                               tts: &[tokenstream::TokenTree])
                               -> Box<base::MacResult + 'cx> {

    match parse_args(ecx, sp, tts) {
        Some((efmt, args, names)) => {
            MacEager::expr(expand_preparsed_format_args(ecx, sp, efmt, args, names))
        }
        None => DummyResult::expr(sp),
    }
}

/// Take the various parts of `format_args!(efmt, args..., name=names...)`
/// and construct the appropriate formatting expression.
pub fn expand_preparsed_format_args(ecx: &mut ExtCtxt,
                                    sp: Span,
                                    efmt: P<ast::Expr>,
                                    args: Vec<P<ast::Expr>>,
                                    names: HashMap<String, usize>)
                                    -> P<ast::Expr> {
    // NOTE: this verbose way of initializing `Vec<Vec<ArgumentType>>` is because
    // `ArgumentType` does not derive `Clone`.
    let arg_types: Vec<_> = (0..args.len()).map(|_| Vec::new()).collect();
    let arg_unique_types: Vec<_> = (0..args.len()).map(|_| Vec::new()).collect();
    let macsp = ecx.call_site();
    let msg = "format argument must be a string literal.";
    let fmt = match expr_to_spanned_string(ecx, efmt, msg) {
        Some(fmt) => fmt,
        None => return DummyResult::raw_expr(sp),
    };

    let mut cx = Context {
        ecx: ecx,
        args: args,
        arg_types: arg_types,
        arg_unique_types: arg_unique_types,
        names: names,
        curarg: 0,
        arg_index_map: Vec::new(),
        count_args: Vec::new(),
        count_positions: HashMap::new(),
        count_positions_count: 0,
        count_args_index_offset: 0,
        literal: String::new(),
        pieces: Vec::new(),
        str_pieces: Vec::new(),
        all_pieces_simple: true,
        macsp: macsp,
        fmtsp: fmt.span,
    };

    let fmt_str = &*fmt.node.0.as_str();
    let mut parser = parse::Parser::new(fmt_str);
    let mut pieces = vec![];

    loop {
        match parser.next() {
            Some(mut piece) => {
                if !parser.errors.is_empty() {
                    break;
                }
                cx.verify_piece(&piece);
                cx.resolve_name_inplace(&mut piece);
                pieces.push(piece);
            }
            None => break,
        }
    }

    cx.build_index_map();

    let mut arg_index_consumed = vec![0usize; cx.arg_index_map.len()];
    for piece in pieces {
        if let Some(piece) = cx.trans_piece(&piece, &mut arg_index_consumed) {
            let s = cx.trans_literal_string();
            cx.str_pieces.push(s);
            cx.pieces.push(piece);
        }
    }

    if !parser.errors.is_empty() {
        let (err, note) = parser.errors.remove(0);
        let mut e = cx.ecx.struct_span_err(cx.fmtsp, &format!("invalid format string: {}", err));
        if let Some(note) = note {
            e.note(&note);
        }
        e.emit();
        return DummyResult::raw_expr(sp);
    }
    if !cx.literal.is_empty() {
        let s = cx.trans_literal_string();
        cx.str_pieces.push(s);
    }

    // Make sure that all arguments were used and all arguments have types.
    let num_pos_args = cx.args.len() - cx.names.len();
    let mut errs = vec![];
    for (i, ty) in cx.arg_types.iter().enumerate() {
        if ty.len() == 0 {
            if cx.count_positions.contains_key(&i) {
                continue;
            }
            let msg = if i >= num_pos_args {
                // named argument
                "named argument never used"
            } else {
                // positional argument
                "argument never used"
            };
            errs.push((cx.args[i].span, msg));
        }
    }
    if errs.len() > 0 {
        let args_used = cx.arg_types.len() - errs.len();
        let args_unused = errs.len();

        let mut diag = {
            if errs.len() == 1 {
                let (sp, msg) = errs.into_iter().next().unwrap();
                cx.ecx.struct_span_err(sp, msg)
            } else {
                let mut diag = cx.ecx.struct_span_err(cx.fmtsp,
                    "multiple unused formatting arguments");
                for (sp, msg) in errs {
                    diag.span_note(sp, msg);
                }
                diag
            }
        };

        // Decide if we want to look for foreign formatting directives.
        if args_used < args_unused {
            use super::format_foreign as foreign;

            // The set of foreign substitutions we've explained.  This prevents spamming the user
            // with `%d should be written as {}` over and over again.
            let mut explained = HashSet::new();

            // Used to ensure we only report translations for *one* kind of foreign format.
            let mut found_foreign = false;

            macro_rules! check_foreign {
                ($kind:ident) => {{
                    let mut show_doc_note = false;

                    for sub in foreign::$kind::iter_subs(fmt_str) {
                        let trn = match sub.translate() {
                            Some(trn) => trn,

                            // If it has no translation, don't call it out specifically.
                            None => continue,
                        };

                        let sub = String::from(sub.as_str());
                        if explained.contains(&sub) {
                            continue;
                        }
                        explained.insert(sub.clone());

                        if !found_foreign {
                            found_foreign = true;
                            show_doc_note = true;
                        }

                        diag.help(&format!("`{}` should be written as `{}`", sub, trn));
                    }

                    if show_doc_note {
                        diag.note(concat!(stringify!($kind), " formatting not supported; see \
                                the documentation for `std::fmt`"));
                    }
                }};
            }

            check_foreign!(printf);
            if !found_foreign {
                check_foreign!(shell);
            }
        }

        diag.emit();
    }

    cx.into_expr()
}
