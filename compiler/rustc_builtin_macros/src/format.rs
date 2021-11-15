use ArgumentType::*;
use Position::*;

use rustc_ast as ast;
use rustc_ast::ptr::P;
use rustc_ast::tokenstream::TokenStream;
use rustc_ast::{token, BlockCheckMode, UnsafeSource};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::{pluralize, Applicability, DiagnosticBuilder};
use rustc_expand::base::{self, *};
use rustc_parse_format as parse;
use rustc_span::symbol::{sym, Ident, Symbol};
use rustc_span::{MultiSpan, Span};

use std::borrow::Cow;
use std::collections::hash_map::Entry;

#[derive(PartialEq)]
enum ArgumentType {
    Placeholder(&'static str),
    Count,
}

enum Position {
    Exact(usize),
    Named(Symbol),
}

struct Context<'a, 'b> {
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
    names: FxHashMap<Symbol, usize>,

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
    /// mapping in `build_piece`. In effect this can be seen as a "flattened"
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
    count_positions: FxHashMap<usize, usize>,
    /// Number of count slots assigned.
    count_positions_count: usize,

    /// Current position of the implicit positional arg pointer, as if it
    /// still existed in this phase of processing.
    /// Used only for `all_pieces_simple` tracking in `build_piece`.
    curarg: usize,
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

            match p.expect(&token::Comma) {
                Err(mut err) => {
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
                Ok(recovered) => {
                    assert!(recovered);
                }
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
    fn resolve_name_inplace(&self, p: &mut parse::Piece<'_>) {
        // NOTE: the `unwrap_or` branch is needed in case of invalid format
        // arguments, e.g., `format_args!("{foo}")`.
        let lookup = |s: Symbol| *self.names.get(&s).unwrap_or(&0);

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
    fn verify_piece(&mut self, p: &parse::Piece<'_>) {
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
                    parse::ArgumentIs(i) | parse::ArgumentImplicitlyIs(i) => Exact(i),
                    parse::ArgumentNamed(s) => Named(s),
                };

                let ty = Placeholder(match arg.format.ty {
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
                });
                self.verify_arg_type(pos, ty);
                self.curpiece += 1;
            }
        }
    }

    fn verify_count(&mut self, c: parse::Count) {
        match c {
            parse::CountImplied | parse::CountIs(..) => {}
            parse::CountIsParam(i) => {
                self.verify_arg_type(Exact(i), Count);
            }
            parse::CountIsName(s) => {
                self.verify_arg_type(Named(s), Count);
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

        let count = self.pieces.len()
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
                        let count = self.pieces.len()
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

    /// Actually verifies and tracks a given format placeholder
    /// (a.k.a. argument).
    fn verify_arg_type(&mut self, arg: Position, ty: ArgumentType) {
        match arg {
            Exact(arg) => {
                if self.args.len() <= arg {
                    self.invalid_refs.push((arg, self.curpiece));
                    return;
                }
                match ty {
                    Placeholder(_) => {
                        // record every (position, type) combination only once
                        let seen_ty = &mut self.arg_unique_types[arg];
                        let i = seen_ty.iter().position(|x| *x == ty).unwrap_or_else(|| {
                            let i = seen_ty.len();
                            seen_ty.push(ty);
                            i
                        });
                        self.arg_types[arg].push(i);
                    }
                    Count => {
                        if let Entry::Vacant(e) = self.count_positions.entry(arg) {
                            let i = self.count_positions_count;
                            e.insert(i);
                            self.count_args.push(Exact(arg));
                            self.count_positions_count += 1;
                        }
                    }
                }
            }

            Named(name) => {
                match self.names.get(&name) {
                    Some(&idx) => {
                        // Treat as positional arg.
                        self.verify_arg_type(Exact(idx), ty)
                    }
                    None => {
                        // For the moment capturing variables from format strings expanded from macros is
                        // disabled (see RFC #2795)
                        if self.is_literal {
                            // Treat this name as a variable to capture from the surrounding scope
                            let idx = self.args.len();
                            self.arg_types.push(Vec::new());
                            self.arg_unique_types.push(Vec::new());
                            let span = if self.is_literal {
                                *self.arg_spans.get(self.curpiece).unwrap_or(&self.fmtsp)
                            } else {
                                self.fmtsp
                            };
                            self.args.push(self.ecx.expr_ident(span, Ident::new(name, span)));
                            self.names.insert(name, idx);
                            self.verify_arg_type(Exact(idx), ty)
                        } else {
                            let msg = format!("there is no argument named `{}`", name);
                            let sp = if self.is_literal {
                                *self.arg_spans.get(self.curpiece).unwrap_or(&self.fmtsp)
                            } else {
                                self.fmtsp
                            };
                            let mut err = self.ecx.struct_span_err(sp, &msg[..]);

                            err.note(&format!(
                                "did you intend to capture a variable `{}` from \
                                 the surrounding scope?",
                                name
                            ));
                            err.note(
                                "to avoid ambiguity, `format_args!` cannot capture variables \
                                 when the format string is expanded from a macro",
                            );

                            err.emit();
                        }
                    }
                }
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
            let arg_types = &self.arg_types[i];
            let arg_offsets = arg_types.iter().map(|offset| sofar + *offset).collect::<Vec<_>>();
            self.arg_index_map.push(arg_offsets);
            sofar += self.arg_unique_types[i].len();
        }

        // Record starting index for counts, which appear just after arguments
        self.count_args_index_offset = sofar;
    }

    fn rtpath(ecx: &ExtCtxt<'_>, s: Symbol) -> Vec<Ident> {
        ecx.std_path(&[sym::fmt, sym::rt, sym::v1, s])
    }

    fn build_count(&self, c: parse::Count) -> P<ast::Expr> {
        let sp = self.macsp;
        let count = |c, arg| {
            let mut path = Context::rtpath(self.ecx, sym::Count);
            path.push(Ident::new(c, sp));
            match arg {
                Some(arg) => self.ecx.expr_call_global(sp, path, vec![arg]),
                None => self.ecx.expr_path(self.ecx.path_global(sp, path)),
            }
        };
        match c {
            parse::CountIs(i) => count(sym::Is, Some(self.ecx.expr_usize(sp, i))),
            parse::CountIsParam(i) => {
                // This needs mapping too, as `i` is referring to a macro
                // argument. If `i` is not found in `count_positions` then
                // the error had already been emitted elsewhere.
                let i = self.count_positions.get(&i).cloned().unwrap_or(0)
                    + self.count_args_index_offset;
                count(sym::Param, Some(self.ecx.expr_usize(sp, i)))
            }
            parse::CountImplied => count(sym::Implied, None),
            // should never be the case, names are already resolved
            parse::CountIsName(_) => panic!("should never happen"),
        }
    }

    /// Build a literal expression from the accumulated string literals
    fn build_literal_string(&mut self) -> P<ast::Expr> {
        let sp = self.fmtsp;
        let s = Symbol::intern(&self.literal);
        self.literal.clear();
        self.ecx.expr_str(sp, s)
    }

    /// Builds a static `rt::Argument` from a `parse::Piece` or append
    /// to the `literal` string.
    fn build_piece(
        &mut self,
        piece: &parse::Piece<'a>,
        arg_index_consumed: &mut Vec<usize>,
    ) -> Option<P<ast::Expr>> {
        let sp = self.macsp;
        match *piece {
            parse::String(s) => {
                self.literal.push_str(s);
                None
            }
            parse::NextArgument(ref arg) => {
                // Build the position
                let pos = {
                    match arg.position {
                        parse::ArgumentIs(i) | parse::ArgumentImplicitlyIs(i) => {
                            // Map to index in final generated argument array
                            // in case of multiple types specified
                            let arg_idx = match arg_index_consumed.get_mut(i) {
                                None => 0, // error already emitted elsewhere
                                Some(offset) => {
                                    let idx_map = &self.arg_index_map[i];
                                    // unwrap_or branch: error already emitted elsewhere
                                    let arg_idx = *idx_map.get(*offset).unwrap_or(&0);
                                    *offset += 1;
                                    arg_idx
                                }
                            };
                            self.ecx.expr_usize(sp, arg_idx)
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
                        precision_span: None,
                        width: parse::CountImplied,
                        width_span: None,
                        ty: arg.format.ty,
                        ty_span: arg.format.ty_span,
                    },
                };

                let fill = arg.format.fill.unwrap_or(' ');

                let pos_simple = arg.position.index() == simple_arg.position.index();

                if arg.format.precision_span.is_some() || arg.format.width_span.is_some() {
                    self.arg_with_formatting.push(arg.format);
                }
                if !pos_simple || arg.format != simple_arg.format || fill != ' ' {
                    self.all_pieces_simple = false;
                }

                // Build the format
                let fill = self.ecx.expr_lit(sp, ast::LitKind::Char(fill));
                let align = |name| {
                    let mut p = Context::rtpath(self.ecx, sym::Alignment);
                    p.push(Ident::new(name, sp));
                    self.ecx.path_global(sp, p)
                };
                let align = match arg.format.align {
                    parse::AlignLeft => align(sym::Left),
                    parse::AlignRight => align(sym::Right),
                    parse::AlignCenter => align(sym::Center),
                    parse::AlignUnknown => align(sym::Unknown),
                };
                let align = self.ecx.expr_path(align);
                let flags = self.ecx.expr_u32(sp, arg.format.flags);
                let prec = self.build_count(arg.format.precision);
                let width = self.build_count(arg.format.width);
                let path = self.ecx.path_global(sp, Context::rtpath(self.ecx, sym::FormatSpec));
                let fmt = self.ecx.expr_struct(
                    sp,
                    path,
                    vec![
                        self.ecx.field_imm(sp, Ident::new(sym::fill, sp), fill),
                        self.ecx.field_imm(sp, Ident::new(sym::align, sp), align),
                        self.ecx.field_imm(sp, Ident::new(sym::flags, sp), flags),
                        self.ecx.field_imm(sp, Ident::new(sym::precision, sp), prec),
                        self.ecx.field_imm(sp, Ident::new(sym::width, sp), width),
                    ],
                );

                let path = self.ecx.path_global(sp, Context::rtpath(self.ecx, sym::Argument));
                Some(self.ecx.expr_struct(
                    sp,
                    path,
                    vec![
                        self.ecx.field_imm(sp, Ident::new(sym::position, sp), pos),
                        self.ecx.field_imm(sp, Ident::new(sym::format, sp), fmt),
                    ],
                ))
            }
        }
    }

    /// Actually builds the expression which the format_args! block will be
    /// expanded to.
    fn into_expr(self) -> P<ast::Expr> {
        let mut args = Vec::with_capacity(
            self.arg_unique_types.iter().map(|v| v.len()).sum::<usize>() + self.count_args.len(),
        );
        let mut heads = Vec::with_capacity(self.args.len());

        // First, build up the static array which will become our precompiled
        // format "string"
        let pieces = self.ecx.expr_vec_slice(self.fmtsp, self.str_pieces);

        // Before consuming the expressions, we have to remember spans for
        // count arguments as they are now generated separate from other
        // arguments, hence have no access to the `P<ast::Expr>`'s.
        let spans_pos: Vec<_> = self.args.iter().map(|e| e.span).collect();

        // Right now there is a bug such that for the expression:
        //      foo(bar(&1))
        // the lifetime of `1` doesn't outlast the call to `bar`, so it's not
        // valid for the call to `foo`. To work around this all arguments to the
        // format! string are shoved into locals. Furthermore, we shove the address
        // of each variable because we don't want to move out of the arguments
        // passed to this function.
        for (i, e) in self.args.into_iter().enumerate() {
            for arg_ty in self.arg_unique_types[i].iter() {
                args.push(Context::format_arg(self.ecx, self.macsp, e.span, arg_ty, i));
            }
            heads.push(self.ecx.expr_addr_of(e.span, e));
        }
        for pos in self.count_args {
            let index = match pos {
                Exact(i) => i,
                _ => panic!("should never happen"),
            };
            let span = spans_pos[index];
            args.push(Context::format_arg(self.ecx, self.macsp, span, &Count, index));
        }

        let args_array = self.ecx.expr_vec(self.macsp, args);

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
        let args_match = {
            let pat = self.ecx.pat_ident(self.macsp, Ident::new(sym::_args, self.macsp));
            let arm = self.ecx.arm(self.macsp, pat, args_array);
            let head = self.ecx.expr(self.macsp, ast::ExprKind::Tup(heads));
            self.ecx.expr_match(self.macsp, head, vec![arm])
        };

        let args_slice = self.ecx.expr_addr_of(self.macsp, args_match);

        // Now create the fmt::Arguments struct with all our locals we created.
        let (fn_name, fn_args) = if self.all_pieces_simple {
            ("new_v1", vec![pieces, args_slice])
        } else {
            // Build up the static array which will store our precompiled
            // nonstandard placeholders, if there are any.
            let fmt = self.ecx.expr_vec_slice(self.macsp, self.pieces);

            let path = self.ecx.std_path(&[sym::fmt, sym::UnsafeArg, sym::new]);
            let unsafe_arg = self.ecx.expr_call_global(self.macsp, path, Vec::new());
            let unsafe_expr = self.ecx.expr_block(P(ast::Block {
                stmts: vec![self.ecx.stmt_expr(unsafe_arg)],
                id: ast::DUMMY_NODE_ID,
                rules: BlockCheckMode::Unsafe(UnsafeSource::CompilerGenerated),
                span: self.macsp,
                tokens: None,
                could_be_bare_literal: false,
            }));

            ("new_v1_formatted", vec![pieces, args_slice, fmt, unsafe_expr])
        };

        let path = self.ecx.std_path(&[sym::fmt, sym::Arguments, Symbol::intern(fn_name)]);
        self.ecx.expr_call_global(self.macsp, path, fn_args)
    }

    fn format_arg(
        ecx: &ExtCtxt<'_>,
        macsp: Span,
        mut sp: Span,
        ty: &ArgumentType,
        arg_index: usize,
    ) -> P<ast::Expr> {
        sp = ecx.with_def_site_ctxt(sp);
        let arg = ecx.expr_ident(sp, Ident::new(sym::_args, sp));
        let arg = ecx.expr(sp, ast::ExprKind::Field(arg, Ident::new(sym::integer(arg_index), sp)));
        let trait_ = match *ty {
            Placeholder(trait_) if trait_ == "<invalid>" => return DummyResult::raw_expr(sp, true),
            Placeholder(trait_) => trait_,
            Count => {
                let path = ecx.std_path(&[sym::fmt, sym::ArgumentV1, sym::from_usize]);
                return ecx.expr_call_global(macsp, path, vec![arg]);
            }
        };

        let path = ecx.std_path(&[sym::fmt, Symbol::intern(trait_), sym::fmt]);
        let format_fn = ecx.path_global(sp, path);
        let path = ecx.std_path(&[sym::fmt, sym::ArgumentV1, sym::new]);
        ecx.expr_call_global(macsp, path, vec![arg, ecx.expr_path(format_fn)])
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
    // NOTE: this verbose way of initializing `Vec<Vec<ArgumentType>>` is because
    // `ArgumentType` does not derive `Clone`.
    let arg_types: Vec<_> = (0..args.len()).map(|_| Vec::new()).collect();
    let arg_unique_types: Vec<_> = (0..args.len()).map(|_| Vec::new()).collect();

    let mut macsp = ecx.call_site();
    macsp = ecx.with_def_site_ctxt(macsp);

    let msg = "format argument must be a string literal";
    let fmt_sp = efmt.span;
    let efmt_kind_is_lit: bool = matches!(efmt.kind, ast::ExprKind::Lit(_));
    let (fmt_str, fmt_style, fmt_span) = match expr_to_spanned_string(ecx, efmt, msg) {
        Ok(mut fmt) if append_newline => {
            fmt.0 = Symbol::intern(&format!("{}\n", fmt.0));
            fmt
        }
        Ok(fmt) => fmt,
        Err(err) => {
            if let Some((mut err, suggested)) = err {
                let sugg_fmt = match args.len() {
                    0 => "{}".to_string(),
                    _ => format!("{}{{}}", "{} ".repeat(args.len())),
                };
                if !suggested {
                    err.span_suggestion(
                        fmt_sp.shrink_to_lo(),
                        "you might be missing a string literal to format with",
                        format!("\"{}\", ", sugg_fmt),
                        Applicability::MaybeIncorrect,
                    );
                }
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
        let sp = if efmt_kind_is_lit {
            fmt_span.from_inner(err.span)
        } else {
            // The format string could be another macro invocation, e.g.:
            //     format!(concat!("abc", "{}"), 4);
            // However, `err.span` is an inner span relative to the *result* of
            // the macro invocation, which is why we would get a nonsensical
            // result calling `fmt_span.from_inner(err.span)` as above, and
            // might even end up inside a multibyte character (issue #86085).
            // Therefore, we conservatively report the error for the entire
            // argument span here.
            fmt_span
        };
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
        args,
        arg_types,
        arg_unique_types,
        names,
        curarg: 0,
        curpiece: 0,
        arg_index_map: Vec::new(),
        count_args: Vec::new(),
        count_positions: FxHashMap::default(),
        count_positions_count: 0,
        count_args_index_offset: 0,
        literal: String::new(),
        pieces: Vec::with_capacity(unverified_pieces.len()),
        str_pieces: Vec::with_capacity(unverified_pieces.len()),
        all_pieces_simple: true,
        macsp,
        fmtsp: fmt_span,
        invalid_refs: Vec::new(),
        arg_spans,
        arg_with_formatting: Vec::new(),
        is_literal: parser.is_literal,
    };

    // This needs to happen *after* the Parser has consumed all pieces to create all the spans
    let pieces = unverified_pieces
        .into_iter()
        .map(|mut piece| {
            cx.verify_piece(&piece);
            cx.resolve_name_inplace(&mut piece);
            piece
        })
        .collect::<Vec<_>>();

    let numbered_position_args = pieces.iter().any(|arg: &parse::Piece<'_>| match *arg {
        parse::String(_) => false,
        parse::NextArgument(arg) => matches!(arg.position, parse::Position::ArgumentIs(_)),
    });

    cx.build_index_map();

    let mut arg_index_consumed = vec![0usize; cx.arg_index_map.len()];

    for piece in pieces {
        if let Some(piece) = cx.build_piece(&piece, &mut arg_index_consumed) {
            let s = cx.build_literal_string();
            cx.str_pieces.push(s);
            cx.pieces.push(piece);
        }
    }

    if !cx.literal.is_empty() {
        let s = cx.build_literal_string();
        cx.str_pieces.push(s);
    }

    if !cx.invalid_refs.is_empty() {
        cx.report_invalid_references(numbered_position_args);
    }

    // Make sure that all arguments were used and all arguments have types.
    let errs = cx
        .arg_types
        .iter()
        .enumerate()
        .filter(|(i, ty)| ty.is_empty() && !cx.count_positions.contains_key(&i))
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
        let args_used = cx.arg_types.len() - errs_len;
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
                        let (trn, success) = match sub.translate() {
                            Ok(trn) => (trn, true),
                            Err(Some(msg)) => (msg, false),

                            // If it has no translation, don't call it out specifically.
                            _ => continue,
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

                            if success {
                                suggestions.push((sp, trn));
                            } else {
                                diag.span_note(
                                    sp,
                                    &format!("format specifiers use curly braces, and {}", trn),
                                );
                            }
                        } else {
                            if success {
                                diag.help(&format!("`{}` should be written as `{}`", sub, trn));
                            } else {
                                diag.note(&format!(
                                    "`{}` should use curly braces, and {}",
                                    sub, trn
                                ));
                            }
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
