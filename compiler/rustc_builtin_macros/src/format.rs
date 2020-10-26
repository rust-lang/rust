#![warn(warnings)]

use rustc_ast as ast;
use rustc_ast::ptr::P;
use rustc_ast::token;
use rustc_ast::tokenstream::TokenStream;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::{pluralize, Applicability, DiagnosticBuilder};
use rustc_expand::base::{self, *};
use rustc_parse_format as parse;
use rustc_span::symbol::{sym, Ident, Symbol};
use rustc_span::{MultiSpan, Span};

use std::borrow::Cow;

enum Position {
    Exact(usize),
    Named(Symbol),
}

use Position::*;

struct Context<'a, 'b> {
    ecx: &'a mut ExtCtxt<'b>,
    /// The macro's call site. References to unstable formatting internals must
    /// use this span to pass the stability checker.
    macsp: Span,
    /// The span of the format string literal.
    fmtsp: Span,
    /// List of parsed argument expressions.
    args: Vec<P<ast::Expr>>,
    /// Flags for each argument whether they are used for anything.
    arg_used: Vec<bool>,
    /// Map from named arguments to their resolved indices.
    names: FxHashMap<Symbol, usize>,
    /// The latest consecutive literal strings, or empty if there weren't any.
    literal: String,
    /// Collection of the compiled `rt::v2::Cmd` commands.
    commands: Vec<P<ast::Expr>>,
    /// Current piece being evaluated, used for error reporting.
    curpiece: usize,
    /// Keep track of invalid references to positional arguments.
    invalid_refs: Vec<(usize, usize)>,
    /// Spans of all the formatting arguments, in order.
    arg_spans: Vec<Span>,
    /// All the formatting arguments that have formatting flags set, in order for diagnostics.
    arg_with_formatting: Vec<parse::FormatSpec<'a>>,
    /// Whether this format string came from a string literal, as opposed to a macro.
    is_literal: bool,
    /// Set to true when trying to use an implicit capture without the feature enabled.
    no_implicit_capture: bool,
    /// Number of `Format` commands in self.commands.
    n_format_commands: usize,
}

/// Parses the arguments from the given list of tokens, returning the diagnostic
/// if there's a parse error so we can continue parsing other format!
/// expressions.
///
/// If parsing succeeds, the return value is:
///
/// ```text
/// Some((fmtstr, parsed arguments, index map for named arguments))
/// ```
fn parse_args<'a>(
    ecx: &mut ExtCtxt<'a>,
    sp: Span,
    tts: TokenStream,
) -> Result<(P<ast::Expr>, Vec<P<ast::Expr>>, FxHashMap<Symbol, usize>), DiagnosticBuilder<'a>> {
    let mut args = Vec::<P<ast::Expr>>::new();
    let mut names = FxHashMap::<Symbol, usize>::default();

    let mut p = ecx.new_parser_from_tts(tts);

    if p.token == token::Eof {
        return Err(ecx.struct_span_err(sp, "requires at least a format string argument"));
    }

    let first_token = &p.token;
    let fmtstr = match first_token.kind {
        token::TokenKind::Literal(token::Lit {
            kind: token::LitKind::Str | token::LitKind::StrRaw(_),
            ..
        }) => {
            // If the first token is a string literal, then a format expression
            // is constructed from it.
            //
            // This allows us to properly handle cases when the first comma
            // after the format string is mistakenly replaced with any operator,
            // which cause the expression parser to eat too much tokens.
            p.parse_literal_maybe_minus()?
        }
        _ => {
            // Otherwise, we fall back to the expression parser.
            p.parse_expr()?
        }
    };

    let mut first = true;
    let mut named = false;

    while p.token != token::Eof {
        if !p.eat(&token::Comma) {
            if first {
                p.clear_expected_tokens();
            }

            // `Parser::expect` tries to recover using the
            // `Parser::unexpected_try_recover` function. This function is able
            // to recover if the expected token is a closing delimiter.
            //
            // As `,` is not a closing delimiter, it will always return an `Err`
            // variant.
            let mut err = p.expect(&token::Comma).unwrap_err();

            match token::TokenKind::Comma.similar_tokens() {
                Some(tks) if tks.contains(&p.token.kind) => {
                    // If a similar token is found, then it may be a typo. We
                    // consider it as a comma, and continue parsing.
                    err.emit();
                    p.bump();
                }
                // Otherwise stop the parsing and return the error.
                _ => return Err(err),
            }
        }
        first = false;
        if p.token == token::Eof {
            break;
        } // accept trailing commas
        match p.token.ident() {
            Some((ident, _)) if p.look_ahead(1, |t| *t == token::Eq) => {
                named = true;
                p.bump();
                p.expect(&token::Eq)?;
                let e = p.parse_expr()?;
                if let Some(prev) = names.get(&ident.name) {
                    ecx.struct_span_err(e.span, &format!("duplicate argument named `{}`", ident))
                        .span_label(args[*prev].span, "previously here")
                        .span_label(e.span, "duplicate argument")
                        .emit();
                    continue;
                }

                // Resolve names into slots early.
                // Since all the positional args are already seen at this point
                // if the input is valid, we can simply append to the positional
                // args. And remember the names.
                let slot = args.len();
                names.insert(ident.name, slot);
                args.push(e);
            }
            _ => {
                let e = p.parse_expr()?;
                if named {
                    let mut err = ecx.struct_span_err(
                        e.span,
                        "positional arguments cannot follow named arguments",
                    );
                    err.span_label(e.span, "positional arguments must be before named arguments");
                    for pos in names.values() {
                        err.span_label(args[*pos].span, "named argument");
                    }
                    err.emit();
                }
                args.push(e);
            }
        }
    }
    Ok((fmtstr, args, names))
}

impl<'a, 'b> Context<'a, 'b> {
    /// Verifies one piece of a parse string, and remembers it if valid.
    /// All errors are not emitted as fatal so we can continue giving errors
    /// about this and possibly other format strings.
    fn verify_piece(&mut self, p: &parse::Piece<'_>) -> Option<&'static str> {
        match *p {
            parse::String(..) => None,
            parse::NextArgument(ref arg) => {
                // width/precision first, if they have implicit positional
                // parameters it makes more sense to consume them first.
                self.verify_count(arg.format.width);
                self.verify_count(arg.format.precision);

                // argument second, if it's an implicit positional parameter
                // it's written second, so it should come after width/precision.
                let pos = match arg.position {
                    parse::ArgumentIs(i) | parse::ArgumentImplicitlyIs(i) => Exact(i),
                    parse::ArgumentNamed(s) => Named(s),
                };

                let ty = match &arg.format.ty[..] {
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
                        let fmtsp = self.fmtsp;
                        let sp = arg.format.ty_span.map(|sp| fmtsp.from_inner(sp));
                        let mut err = self.ecx.struct_span_err(
                            sp.unwrap_or(fmtsp),
                            &format!("unknown format trait `{}`", arg.format.ty),
                        );
                        err.note(
                            "the only appropriate formatting traits are:\n\
                                - ``, which uses the `Display` trait\n\
                                - `?`, which uses the `Debug` trait\n\
                                - `e`, which uses the `LowerExp` trait\n\
                                - `E`, which uses the `UpperExp` trait\n\
                                - `o`, which uses the `Octal` trait\n\
                                - `p`, which uses the `Pointer` trait\n\
                                - `b`, which uses the `Binary` trait\n\
                                - `x`, which uses the `LowerHex` trait\n\
                                - `X`, which uses the `UpperHex` trait",
                        );
                        if let Some(sp) = sp {
                            for (fmt, name) in &[
                                ("", "Display"),
                                ("?", "Debug"),
                                ("e", "LowerExp"),
                                ("E", "UpperExp"),
                                ("o", "Octal"),
                                ("p", "Pointer"),
                                ("b", "Binary"),
                                ("x", "LowerHex"),
                                ("X", "UpperHex"),
                            ] {
                                // FIXME: rustfix (`run-rustfix`) fails to apply suggestions.
                                // > "Cannot replace slice of data that was already replaced"
                                err.tool_only_span_suggestion(
                                    sp,
                                    &format!("use the `{}` trait", name),
                                    (*fmt).to_string(),
                                    Applicability::MaybeIncorrect,
                                );
                            }
                        }
                        err.emit();
                        "<invalid>"
                    }
                };
                self.verify_arg_exists(pos);
                self.curpiece += 1;
                Some(ty)
            }
        }
    }

    fn verify_count(&mut self, c: parse::Count) {
        match c {
            parse::CountImplied | parse::CountIs(..) => {}
            parse::CountIsParam(i) => {
                self.verify_arg_exists(Exact(i));
            }
            parse::CountIsName(s) => {
                self.verify_arg_exists(Named(s));
            }
        }
    }

    fn describe_num_args(&self) -> Cow<'_, str> {
        match self.args.len() {
            0 => "no arguments were given".into(),
            1 => "there is 1 argument".into(),
            x => format!("there are {} arguments", x).into(),
        }
    }

    /// Handle invalid references to positional arguments. Output different
    /// errors for the case where all arguments are positional and for when
    /// there are named arguments or numbered positional arguments in the
    /// format string.
    fn report_invalid_references(&self, numbered_position_args: bool) {
        let mut e;
        let sp = if !self.arg_spans.is_empty() {
            // Point at the formatting arguments.
            MultiSpan::from_spans(self.arg_spans.clone())
        } else {
            MultiSpan::from_span(self.fmtsp)
        };
        let refs =
            self.invalid_refs.iter().map(|(r, pos)| (r.to_string(), self.arg_spans.get(*pos)));

        let mut zero_based_note = false;

        let count = self.n_format_commands
            + self.arg_with_formatting.iter().filter(|fmt| fmt.precision_span.is_some()).count();

        if self.names.is_empty() && !numbered_position_args && count != self.args.len() {
            e = self.ecx.struct_span_err(
                sp,
                &format!(
                    "{} positional argument{} in format string, but {}",
                    count,
                    pluralize!(count),
                    self.describe_num_args(),
                ),
            );
            for arg in &self.args {
                // Point at the arguments that will be formatted.
                e.span_label(arg.span, "");
            }
        } else {
            let (mut refs, spans): (Vec<_>, Vec<_>) = refs.unzip();
            // Avoid `invalid reference to positional arguments 7 and 7 (there is 1 argument)`
            // for `println!("{7:7$}", 1);`
            refs.sort();
            refs.dedup();
            let spans: Vec<_> = spans.into_iter().filter_map(|sp| sp.copied()).collect();
            let sp = if self.arg_spans.is_empty() || spans.is_empty() {
                MultiSpan::from_span(self.fmtsp)
            } else {
                MultiSpan::from_spans(spans)
            };
            let arg_list = if refs.len() == 1 {
                format!("argument {}", refs[0])
            } else {
                let reg = refs.pop().unwrap();
                format!("arguments {head} and {tail}", head = refs.join(", "), tail = reg)
            };

            e = self.ecx.struct_span_err(
                sp,
                &format!(
                    "invalid reference to positional {} ({})",
                    arg_list,
                    self.describe_num_args()
                ),
            );
            zero_based_note = true;
        };

        for fmt in &self.arg_with_formatting {
            if let Some(span) = fmt.precision_span {
                let span = self.fmtsp.from_inner(span);
                match fmt.precision {
                    parse::CountIsParam(pos) if pos > self.args.len() => {
                        e.span_label(
                            span,
                            &format!(
                                "this precision flag expects an `usize` argument at position {}, \
                             but {}",
                                pos,
                                self.describe_num_args(),
                            ),
                        );
                        zero_based_note = true;
                    }
                    parse::CountIsParam(pos) => {
                        let count = self.n_format_commands
                            + self
                                .arg_with_formatting
                                .iter()
                                .filter(|fmt| fmt.precision_span.is_some())
                                .count();
                        e.span_label(span, &format!(
                            "this precision flag adds an extra required argument at position {}, \
                             which is why there {} expected",
                            pos,
                            if count == 1 {
                                "is 1 argument".to_string()
                            } else {
                                format!("are {} arguments", count)
                            },
                        ));
                        if let Some(arg) = self.args.get(pos) {
                            e.span_label(
                                arg.span,
                                "this parameter corresponds to the precision flag",
                            );
                        }
                        zero_based_note = true;
                    }
                    _ => {}
                }
            }
            if let Some(span) = fmt.width_span {
                let span = self.fmtsp.from_inner(span);
                match fmt.width {
                    parse::CountIsParam(pos) if pos > self.args.len() => {
                        e.span_label(
                            span,
                            &format!(
                                "this width flag expects an `usize` argument at position {}, \
                             but {}",
                                pos,
                                self.describe_num_args(),
                            ),
                        );
                        zero_based_note = true;
                    }
                    _ => {}
                }
            }
        }
        if zero_based_note {
            e.note("positional arguments are zero-based");
        }
        if !self.arg_with_formatting.is_empty() {
            e.note(
                "for information about formatting flags, visit \
                    https://doc.rust-lang.org/std/fmt/index.html",
            );
        }

        e.emit();
    }

    fn verify_arg_exists(&mut self, arg: Position) {
        match arg {
            Exact(arg) => {
                if self.args.len() <= arg {
                    self.invalid_refs.push((arg, self.curpiece));
                    return;
                }
            }
            Named(name) => {
                if self.names.get(&name).is_none() {
                    let capture_feature_enabled = self
                        .ecx
                        .ecfg
                        .features
                        .map_or(false, |features| features.format_args_capture);

                    // For the moment capturing variables from format strings expanded from macros is
                    // disabled (see RFC #2795)
                    let can_capture = capture_feature_enabled && self.is_literal;

                    if !can_capture {
                        self.no_implicit_capture = true;

                        let msg = format!("there is no argument named `{}`", name);
                        let sp = if self.is_literal {
                            *self.arg_spans.get(self.curpiece).unwrap_or(&self.fmtsp)
                        } else {
                            self.fmtsp
                        };
                        let mut err = self.ecx.struct_span_err(sp, &msg[..]);

                        if capture_feature_enabled && !self.is_literal {
                            err.note(&format!(
                                "did you intend to capture a variable `{}` from \
                                 the surrounding scope?",
                                name
                            ));
                            err.note(
                                "to avoid ambiguity, `format_args!` cannot capture variables \
                                 when the format string is expanded from a macro",
                            );
                        } else if self.ecx.parse_sess().unstable_features.is_nightly_build() {
                            err.help(&format!(
                                "if you intended to capture `{}` from the surrounding scope, add \
                                 `#![feature(format_args_capture)]` to the crate attributes",
                                name
                            ));
                        }

                        err.emit();
                    }
                }
            }
        }
    }

    fn rtpath(ecx: &ExtCtxt<'_>, s: Symbol) -> Vec<Ident> {
        ecx.std_path(&[sym::fmt, sym::rt, sym::v2, s])
    }

    /// Append a command to be executed at runtime.
    fn append_command(&mut self, cmd: Symbol, args: Vec<P<ast::Expr>>) {
        let path = self.ecx.std_path(&[sym::fmt, sym::rt, sym::v2, sym::Cmd, cmd]);
        self.commands.push(self.ecx.expr_call_global(self.macsp, path, args));
        if cmd == sym::Format {
            self.n_format_commands += 1;
        }
    }

    /// Append a command to be executed at runtime.
    fn append_command_struct(&mut self, cmd: Symbol, args: Vec<ast::Field>) {
        let path = self.ecx.std_path(&[sym::fmt, sym::rt, sym::v2, sym::Cmd, cmd]);
        self.commands.push(self.ecx.expr_struct(
            self.macsp,
            self.ecx.path_global(self.macsp, path),
            args,
        ));
    }

    /// Append a literal string command from the accumulated string literals
    fn finish_literal_string(&mut self) {
        if !self.literal.is_empty() {
            let s = Symbol::intern(&self.literal);
            self.literal.clear();
            self.append_command(sym::Str, vec![self.ecx.expr_str(self.fmtsp, s)]);
        }
    }

    fn build_arg(&mut self, arg: parse::Position) -> P<ast::Expr> {
        match arg {
            parse::ArgumentIs(i) | parse::ArgumentImplicitlyIs(i) => {
                if i < self.args.len() {
                    self.arg_used[i] = true;
                    let name = format!("arg{}", i);
                    let sp = self.args[i].span;
                    self.ecx.expr_ident(sp, Ident::from_str_and_span(&name, self.macsp))
                } else {
                    DummyResult::raw_expr(self.macsp, true)
                }
            }
            parse::ArgumentNamed(n) => {
                if let Some(&i) = self.names.get(&n) {
                    self.build_arg(parse::ArgumentIs(i))
                } else if self.no_implicit_capture {
                    DummyResult::raw_expr(self.macsp, true)
                } else {
                    self.ecx.expr_addr_of(
                        self.macsp,
                        self.ecx.expr_ident(self.macsp, Ident::new(n, self.macsp)),
                    )
                }
            }
        }
    }

    fn build_count_arg(&mut self, arg: parse::Count) -> Option<P<ast::Expr>> {
        match arg {
            parse::CountIs(c) => Some(self.ecx.expr_usize(self.macsp, c)),
            parse::CountIsName(n) => {
                let arg = self.build_arg(parse::ArgumentNamed(n));
                Some(self.ecx.expr_deref(self.macsp, arg))
            }
            parse::CountIsParam(i) => {
                let arg = self.build_arg(parse::ArgumentIs(i));
                Some(self.ecx.expr_deref(self.macsp, arg))
            }
            parse::CountImplied => None,
        }
    }

    /// Append a `rt::v2::Cmd` from a `parse::Piece` or append
    /// to the `literal` string.
    fn append_piece(&mut self, piece: &parse::Piece<'a>, trait_: Option<&'static str>) {
        let sp = self.macsp;
        match piece {
            parse::String(s) => self.literal.push_str(s),
            parse::NextArgument(ref arg) => {
                self.finish_literal_string();

                let arg_expr = self.build_arg(arg.position);

                if arg.format.precision_span.is_some() || arg.format.width_span.is_some() {
                    self.arg_with_formatting.push(arg.format);
                }

                // The SetFlags command, if needed.
                let fill = arg.format.fill.unwrap_or(' ');
                if fill != ' ' || arg.format.flags != 0 || arg.format.align != parse::AlignUnknown {
                    let fill = self.ecx.expr_lit(sp, ast::LitKind::Char(fill));
                    let align = |name| {
                        let mut p = Context::rtpath(self.ecx, sym::Alignment);
                        p.push(Ident::new(name, sp));
                        self.ecx.expr_path(self.ecx.path_global(sp, p))
                    };
                    let align = match arg.format.align {
                        parse::AlignLeft => align(sym::Left),
                        parse::AlignRight => align(sym::Right),
                        parse::AlignCenter => align(sym::Center),
                        parse::AlignUnknown => align(sym::Unknown),
                    };
                    let flags = self.ecx.expr_u32(sp, arg.format.flags);
                    self.append_command_struct(
                        sym::SetFlags,
                        vec![
                            self.ecx.field_imm(sp, Ident::new(sym::fill, sp), fill),
                            self.ecx.field_imm(sp, Ident::new(sym::flags, sp), flags),
                            self.ecx.field_imm(sp, Ident::new(sym::align, sp), align),
                        ],
                    );
                }

                // The SetPrecision command, if needed.
                if let Some(prec) = self.build_count_arg(arg.format.precision) {
                    self.append_command(sym::SetPrecision, vec![prec]);
                }

                // The SetWidth command, if needed.
                if let Some(width) = self.build_count_arg(arg.format.width) {
                    self.append_command(sym::SetWidth, vec![width]);
                }

                // The Format command.
                let trait_ = trait_.expect("missing trait for format argument");
                let expr_span = self.ecx.with_def_site_ctxt(arg_expr.span);
                let format_fn = if trait_ == "<invalid>" {
                    DummyResult::raw_expr(expr_span, true)
                } else {
                    let path = self.ecx.std_path(&[sym::fmt, Symbol::intern(trait_), sym::fmt]);
                    self.ecx.expr_path(self.ecx.path_global(expr_span, path))
                };
                self.append_command(
                    sym::Format,
                    vec![self.ecx.expr_call_global(
                        sp,
                        self.ecx.std_path(&[sym::fmt, sym::rt, sym::v2, sym::Argument, sym::new]),
                        vec![arg_expr, format_fn],
                    )],
                );
            }
        }
    }

    /// Actually builds the expression which the format_args! block will be
    /// expanded to.
    fn into_expr(self) -> P<ast::Expr> {
        let mut pats = Vec::with_capacity(self.args.len());
        let mut heads = Vec::with_capacity(self.args.len());

        let names_pos: Vec<_> = (0..self.args.len())
            .map(|i| Ident::from_str_and_span(&format!("arg{}", i), self.macsp))
            .collect();

        let commands = self.ecx.expr_vec(self.fmtsp, self.commands);

        for (i, e) in self.args.into_iter().enumerate() {
            let name = names_pos[i];
            let span = self.ecx.with_def_site_ctxt(e.span);
            pats.push(self.ecx.pat_ident(span, name));
            heads.push(self.ecx.expr_addr_of(e.span, e));
        }

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
        let pat = self.ecx.pat_tuple(self.macsp, pats);
        let arm = self.ecx.arm(self.macsp, pat, commands);
        let head = self.ecx.expr(self.macsp, ast::ExprKind::Tup(heads));
        let result = self.ecx.expr_match(self.macsp, head, vec![arm]);

        // Now create the fmt::Arguments struct.
        self.ecx.expr_call_global(
            self.macsp,
            Context::rtpath(self.ecx, sym::new),
            vec![self.ecx.expr_addr_of(self.macsp, result)],
        )
    }
}

fn expand_format_args_impl<'cx>(
    ecx: &'cx mut ExtCtxt<'_>,
    mut sp: Span,
    tts: TokenStream,
    nl: bool,
) -> Box<dyn base::MacResult + 'cx> {
    sp = ecx.with_def_site_ctxt(sp);
    match parse_args(ecx, sp, tts) {
        Ok((efmt, args, names)) => {
            MacEager::expr(expand_preparsed_format_args(ecx, sp, efmt, args, names, nl))
        }
        Err(mut err) => {
            err.emit();
            DummyResult::any(sp)
        }
    }
}

pub fn expand_format_args<'cx>(
    ecx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> Box<dyn base::MacResult + 'cx> {
    expand_format_args_impl(ecx, sp, tts, false)
}

pub fn expand_format_args_nl<'cx>(
    ecx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> Box<dyn base::MacResult + 'cx> {
    expand_format_args_impl(ecx, sp, tts, true)
}

/// Take the various parts of `format_args!(efmt, args..., name=names...)`
/// and construct the appropriate formatting expression.
pub fn expand_preparsed_format_args(
    ecx: &mut ExtCtxt<'_>,
    sp: Span,
    efmt: P<ast::Expr>,
    args: Vec<P<ast::Expr>>,
    names: FxHashMap<Symbol, usize>,
    append_newline: bool,
) -> P<ast::Expr> {
    let mut macsp = ecx.call_site();
    macsp = ecx.with_def_site_ctxt(macsp);

    let msg = "format argument must be a string literal";
    let fmt_sp = efmt.span;
    let (fmt_str, fmt_style, fmt_span) = match expr_to_spanned_string(ecx, efmt, msg) {
        Ok(mut fmt) if append_newline => {
            fmt.0 = Symbol::intern(&format!("{}\n", fmt.0));
            fmt
        }
        Ok(fmt) => fmt,
        Err(err) => {
            if let Some(mut err) = err {
                let sugg_fmt = match args.len() {
                    0 => "{}".to_string(),
                    _ => format!("{}{{}}", "{} ".repeat(args.len())),
                };
                err.span_suggestion(
                    fmt_sp.shrink_to_lo(),
                    "you might be missing a string literal to format with",
                    format!("\"{}\", ", sugg_fmt),
                    Applicability::MaybeIncorrect,
                );
                err.emit();
            }
            return DummyResult::raw_expr(sp, true);
        }
    };

    let str_style = match fmt_style {
        ast::StrStyle::Cooked => None,
        ast::StrStyle::Raw(raw) => Some(raw as usize),
    };

    let fmt_str = &fmt_str.as_str(); // for the suggestions below
    let fmt_snippet = ecx.source_map().span_to_snippet(fmt_sp).ok();
    let mut parser = parse::Parser::new(
        fmt_str,
        str_style,
        fmt_snippet,
        append_newline,
        parse::ParseMode::Format,
    );

    let mut unverified_pieces = Vec::new();
    while let Some(piece) = parser.next() {
        if !parser.errors.is_empty() {
            break;
        } else {
            unverified_pieces.push(piece);
        }
    }

    if !parser.errors.is_empty() {
        let err = parser.errors.remove(0);
        let sp = fmt_span.from_inner(err.span);
        let mut e = ecx.struct_span_err(sp, &format!("invalid format string: {}", err.description));
        e.span_label(sp, err.label + " in format string");
        if let Some(note) = err.note {
            e.note(&note);
        }
        if let Some((label, span)) = err.secondary_label {
            let sp = fmt_span.from_inner(span);
            e.span_label(sp, label);
        }
        e.emit();
        return DummyResult::raw_expr(sp, true);
    }

    let arg_spans = parser.arg_places.iter().map(|span| fmt_span.from_inner(*span)).collect();

    let named_pos: FxHashSet<usize> = names.values().cloned().collect();

    let mut cx = Context {
        ecx,
        macsp,
        fmtsp: fmt_span,
        arg_used: vec![false; args.len()],
        args,
        names,
        curpiece: 0,
        literal: String::new(),
        commands: Vec::with_capacity(unverified_pieces.len()),
        invalid_refs: Vec::new(),
        arg_spans,
        arg_with_formatting: Vec::new(),
        is_literal: parser.is_literal,
        no_implicit_capture: false,
        n_format_commands: 0,
    };

    // This needs to happen *after* the Parser has consumed all pieces to create all the spans
    for piece in &unverified_pieces {
        let trait_ = cx.verify_piece(piece);
        cx.append_piece(&piece, trait_);
    }
    cx.finish_literal_string();

    let pieces = unverified_pieces;

    let numbered_position_args = pieces.iter().any(|arg: &parse::Piece<'_>| match *arg {
        parse::String(_) => false,
        parse::NextArgument(arg) => match arg.position {
            parse::Position::ArgumentIs(_) => true,
            _ => false,
        },
    });

    if !cx.invalid_refs.is_empty() {
        cx.report_invalid_references(numbered_position_args);
    }

    // Make sure that all arguments were used and all arguments have types.
    let errs = cx
        .arg_used
        .iter()
        .enumerate()
        .filter(|(_, &used)| !used)
        .map(|(i, _)| {
            let msg = if named_pos.contains(&i) {
                // named argument
                "named argument never used"
            } else {
                // positional argument
                "argument never used"
            };
            (cx.args[i].span, msg)
        })
        .collect::<Vec<_>>();

    let errs_len = errs.len();
    if !errs.is_empty() {
        let args_used = cx.arg_used.len() - errs_len;
        let args_unused = errs_len;

        let mut diag = {
            if let [(sp, msg)] = &errs[..] {
                let mut diag = cx.ecx.struct_span_err(*sp, *msg);
                diag.span_label(*sp, *msg);
                diag
            } else {
                let mut diag = cx.ecx.struct_span_err(
                    errs.iter().map(|&(sp, _)| sp).collect::<Vec<Span>>(),
                    "multiple unused formatting arguments",
                );
                diag.span_label(cx.fmtsp, "multiple missing formatting specifiers");
                for (sp, msg) in errs {
                    diag.span_label(sp, msg);
                }
                diag
            }
        };

        // Used to ensure we only report translations for *one* kind of foreign format.
        let mut found_foreign = false;
        // Decide if we want to look for foreign formatting directives.
        if args_used < args_unused {
            use super::format_foreign as foreign;

            // The set of foreign substitutions we've explained.  This prevents spamming the user
            // with `%d should be written as {}` over and over again.
            let mut explained = FxHashSet::default();

            macro_rules! check_foreign {
                ($kind:ident) => {{
                    let mut show_doc_note = false;

                    let mut suggestions = vec![];
                    // account for `"` and account for raw strings `r#`
                    let padding = str_style.map(|i| i + 2).unwrap_or(1);
                    for sub in foreign::$kind::iter_subs(fmt_str, padding) {
                        let trn = match sub.translate() {
                            Some(trn) => trn,

                            // If it has no translation, don't call it out specifically.
                            None => continue,
                        };

                        let pos = sub.position();
                        let sub = String::from(sub.as_str());
                        if explained.contains(&sub) {
                            continue;
                        }
                        explained.insert(sub.clone());

                        if !found_foreign {
                            found_foreign = true;
                            show_doc_note = true;
                        }

                        if let Some(inner_sp) = pos {
                            let sp = fmt_sp.from_inner(inner_sp);
                            suggestions.push((sp, trn));
                        } else {
                            diag.help(&format!("`{}` should be written as `{}`", sub, trn));
                        }
                    }

                    if show_doc_note {
                        diag.note(concat!(
                            stringify!($kind),
                            " formatting not supported; see the documentation for `std::fmt`",
                        ));
                    }
                    if suggestions.len() > 0 {
                        diag.multipart_suggestion(
                            "format specifiers use curly braces",
                            suggestions,
                            Applicability::MachineApplicable,
                        );
                    }
                }};
            }

            check_foreign!(printf);
            if !found_foreign {
                check_foreign!(shell);
            }
        }
        if !found_foreign && errs_len == 1 {
            diag.span_label(cx.fmtsp, "formatting specifier missing");
        }

        diag.emit();
    }

    cx.into_expr()
}
