#![feature(let_chains)]
#![feature(proc_macro_span)]
#![allow(clippy::needless_if, dead_code)]

extern crate proc_macro;

use core::mem;
use proc_macro::token_stream::IntoIter;
use proc_macro::Delimiter::{self, Brace, Parenthesis};
use proc_macro::Spacing::{self, Alone, Joint};
use proc_macro::{Group, Ident, Literal, Punct, Span, TokenStream, TokenTree as TT};

type Result<T> = core::result::Result<T, TokenStream>;

/// Make a `compile_error!` pointing to the given span.
fn make_error(msg: &str, span: Span) -> TokenStream {
    TokenStream::from_iter([
        TT::Ident(Ident::new("compile_error", span)),
        TT::Punct(punct_with_span('!', Alone, span)),
        TT::Group({
            let mut msg = Literal::string(msg);
            msg.set_span(span);
            group_with_span(Parenthesis, TokenStream::from_iter([TT::Literal(msg)]), span)
        }),
    ])
}

fn expect_tt<T>(tt: Option<TT>, f: impl FnOnce(TT) -> Option<T>, expected: &str, span: Span) -> Result<T> {
    match tt {
        None => Err(make_error(
            &format!("unexpected end of input, expected {expected}"),
            span,
        )),
        Some(tt) => {
            let span = tt.span();
            match f(tt) {
                Some(x) => Ok(x),
                None => Err(make_error(&format!("unexpected token, expected {expected}"), span)),
            }
        },
    }
}

fn punct_with_span(c: char, spacing: Spacing, span: Span) -> Punct {
    let mut p = Punct::new(c, spacing);
    p.set_span(span);
    p
}

fn group_with_span(delimiter: Delimiter, stream: TokenStream, span: Span) -> Group {
    let mut g = Group::new(delimiter, stream);
    g.set_span(span);
    g
}

/// Token used to escape the following token from the macro's span rules.
const ESCAPE_CHAR: char = '$';

/// Takes a single token followed by a sequence of tokens. Returns the sequence of tokens with their
/// span set to that of the first token. Tokens may be escaped with either `#ident` or `#(tokens)`.
#[proc_macro]
pub fn with_span(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();
    let span = iter.next().unwrap().span();
    let mut res = TokenStream::new();
    if let Err(e) = write_with_span(span, iter, &mut res) {
        e
    } else {
        res
    }
}

/// Takes a sequence of tokens and return the tokens with the span set such that they appear to be
/// from an external macro. Tokens may be escaped with either `#ident` or `#(tokens)`.
#[proc_macro]
pub fn external(input: TokenStream) -> TokenStream {
    let mut res = TokenStream::new();
    if let Err(e) = write_with_span(Span::mixed_site(), input.into_iter(), &mut res) {
        e
    } else {
        res
    }
}

/// Copies all the tokens, replacing all their spans with the given span. Tokens can be escaped
/// either by `#ident` or `#(tokens)`.
fn write_with_span(s: Span, mut input: IntoIter, out: &mut TokenStream) -> Result<()> {
    while let Some(tt) = input.next() {
        match tt {
            TT::Punct(p) if p.as_char() == ESCAPE_CHAR => {
                expect_tt(
                    input.next(),
                    |tt| match tt {
                        tt @ (TT::Ident(_) | TT::Literal(_)) => {
                            out.extend([tt]);
                            Some(())
                        },
                        TT::Punct(mut p) if p.as_char() == ESCAPE_CHAR => {
                            p.set_span(s);
                            out.extend([TT::Punct(p)]);
                            Some(())
                        },
                        TT::Group(g) if g.delimiter() == Parenthesis => {
                            out.extend([TT::Group(group_with_span(Delimiter::None, g.stream(), g.span()))]);
                            Some(())
                        },
                        _ => None,
                    },
                    "an ident, a literal, or parenthesized tokens",
                    p.span(),
                )?;
            },
            TT::Group(g) => {
                let mut stream = TokenStream::new();
                write_with_span(s, g.stream().into_iter(), &mut stream)?;
                out.extend([TT::Group(group_with_span(g.delimiter(), stream, s))]);
            },
            mut tt => {
                tt.set_span(s);
                out.extend([tt]);
            },
        }
    }
    Ok(())
}

/// Within the item this attribute is attached to, an `inline!` macro is available which expands the
/// contained tokens as though they came from a macro expansion.
///
/// Within the `inline!` macro, any token preceded by `$` is passed as though it were an argument
/// with an automatically chosen fragment specifier. `$ident` will be passed as `ident`, `$1` or
/// `$"literal"` will be passed as `literal`, `$'lt` will be passed as `lifetime`, and `$(...)` will
/// pass the contained tokens as a `tt` sequence (the wrapping parenthesis are removed). If another
/// specifier is required it can be specified within parenthesis like `$(@expr ...)`. This will
/// expand the remaining tokens as a single argument.
///
/// Multiple `inline!` macros may be nested within each other. This will expand as nested macro
/// calls. However, any arguments will be passed as though they came from the outermost context.
#[proc_macro_attribute]
pub fn inline_macros(args: TokenStream, input: TokenStream) -> TokenStream {
    let mut args = args.into_iter();
    let mac_name = match args.next() {
        Some(TT::Ident(name)) => Some(name),
        Some(tt) => {
            return make_error(
                "unexpected argument, expected either an ident or no arguments",
                tt.span(),
            );
        },
        None => None,
    };
    if let Some(tt) = args.next() {
        return make_error(
            "unexpected argument, expected either an ident or no arguments",
            tt.span(),
        );
    };

    let mac_name = if let Some(mac_name) = mac_name {
        Ident::new(&format!("__inline_mac_{mac_name}"), Span::call_site())
    } else {
        let mut input = match LookaheadIter::new(input.clone().into_iter()) {
            Some(x) => x,
            None => return input,
        };
        loop {
            match input.next() {
                None => break Ident::new("__inline_mac", Span::call_site()),
                Some(TT::Ident(kind)) => match &*kind.to_string() {
                    "impl" => break Ident::new("__inline_mac_impl", Span::call_site()),
                    kind @ ("struct" | "enum" | "union" | "fn" | "mod" | "trait" | "type" | "const" | "static") => {
                        if let TT::Ident(name) = &input.tt {
                            break Ident::new(&format!("__inline_mac_{kind}_{name}"), Span::call_site());
                        } else {
                            break Ident::new(&format!("__inline_mac_{kind}"), Span::call_site());
                        }
                    },
                    _ => {},
                },
                _ => {},
            }
        }
    };

    let mut expander = Expander::default();
    let mut mac = MacWriter::new(mac_name);
    if let Err(e) = expander.expand(input.into_iter(), &mut mac) {
        return e;
    }
    let mut out = TokenStream::new();
    mac.finish(&mut out);
    out.extend(expander.expn);
    out
}

/// Wraps a `TokenStream` iterator with a single token lookahead.
struct LookaheadIter {
    tt: TT,
    iter: IntoIter,
}
impl LookaheadIter {
    fn new(mut iter: IntoIter) -> Option<Self> {
        iter.next().map(|tt| Self { tt, iter })
    }

    /// Get's the lookahead token, replacing it with the next token in the stream.
    /// Note: If there isn't a next token, this will not return the lookahead token.
    fn next(&mut self) -> Option<TT> {
        self.iter.next().map(|tt| mem::replace(&mut self.tt, tt))
    }
}

/// Builds the macro used to implement all the `inline!` macro calls.
struct MacWriter {
    name: Ident,
    macros: TokenStream,
    next_idx: usize,
}
impl MacWriter {
    fn new(name: Ident) -> Self {
        Self {
            name,
            macros: TokenStream::new(),
            next_idx: 0,
        }
    }

    /// Inserts a new `inline!` call.
    fn insert(&mut self, name_span: Span, bang_span: Span, body: Group, expander: &mut Expander) -> Result<()> {
        let idx = self.next_idx;
        self.next_idx += 1;

        let mut inner = Expander::for_arm(idx);
        inner.expand(body.stream().into_iter(), self)?;
        let new_arm = inner.arm.unwrap();

        self.macros.extend([
            TT::Group(Group::new(Parenthesis, new_arm.args_def)),
            TT::Punct(Punct::new('=', Joint)),
            TT::Punct(Punct::new('>', Alone)),
            TT::Group(Group::new(Parenthesis, inner.expn)),
            TT::Punct(Punct::new(';', Alone)),
        ]);

        expander.expn.extend([
            TT::Ident({
                let mut name = self.name.clone();
                name.set_span(name_span);
                name
            }),
            TT::Punct(punct_with_span('!', Alone, bang_span)),
        ]);
        let mut call_body = TokenStream::from_iter([TT::Literal(Literal::usize_unsuffixed(idx))]);
        if let Some(arm) = expander.arm.as_mut() {
            if !new_arm.args.is_empty() {
                arm.add_sub_args(new_arm.args, &mut call_body);
            }
        } else {
            call_body.extend(new_arm.args);
        }
        let mut g = Group::new(body.delimiter(), call_body);
        g.set_span(body.span());
        expander.expn.extend([TT::Group(g)]);
        Ok(())
    }

    /// Creates the macro definition.
    fn finish(self, out: &mut TokenStream) {
        if self.next_idx != 0 {
            out.extend([
                TT::Ident(Ident::new("macro_rules", Span::call_site())),
                TT::Punct(Punct::new('!', Alone)),
                TT::Ident(self.name),
                TT::Group(Group::new(Brace, self.macros)),
            ])
        }
    }
}

struct MacroArm {
    args_def: TokenStream,
    args: Vec<TT>,
}
impl MacroArm {
    fn add_single_arg_def(&mut self, kind: &str, dollar_span: Span, arg_span: Span, out: &mut TokenStream) {
        let mut name = Ident::new(&format!("_{}", self.args.len()), Span::call_site());
        self.args_def.extend([
            TT::Punct(Punct::new('$', Alone)),
            TT::Ident(name.clone()),
            TT::Punct(Punct::new(':', Alone)),
            TT::Ident(Ident::new(kind, Span::call_site())),
        ]);
        name.set_span(arg_span);
        out.extend([TT::Punct(punct_with_span('$', Alone, dollar_span)), TT::Ident(name)]);
    }

    fn add_parenthesized_arg_def(&mut self, kind: Ident, dollar_span: Span, arg_span: Span, out: &mut TokenStream) {
        let mut name = Ident::new(&format!("_{}", self.args.len()), Span::call_site());
        self.args_def.extend([TT::Group(Group::new(
            Parenthesis,
            TokenStream::from_iter([
                TT::Punct(Punct::new('$', Alone)),
                TT::Ident(name.clone()),
                TT::Punct(Punct::new(':', Alone)),
                TT::Ident(kind),
            ]),
        ))]);
        name.set_span(arg_span);
        out.extend([TT::Punct(punct_with_span('$', Alone, dollar_span)), TT::Ident(name)]);
    }

    fn add_multi_arg_def(&mut self, dollar_span: Span, arg_span: Span, out: &mut TokenStream) {
        let mut name = Ident::new(&format!("_{}", self.args.len()), Span::call_site());
        self.args_def.extend([TT::Group(Group::new(
            Parenthesis,
            TokenStream::from_iter([
                TT::Punct(Punct::new('$', Alone)),
                TT::Group(Group::new(
                    Parenthesis,
                    TokenStream::from_iter([
                        TT::Punct(Punct::new('$', Alone)),
                        TT::Ident(name.clone()),
                        TT::Punct(Punct::new(':', Alone)),
                        TT::Ident(Ident::new("tt", Span::call_site())),
                    ]),
                )),
                TT::Punct(Punct::new('*', Alone)),
            ]),
        ))]);
        name.set_span(arg_span);
        out.extend([
            TT::Punct(punct_with_span('$', Alone, dollar_span)),
            TT::Group(group_with_span(
                Parenthesis,
                TokenStream::from_iter([TT::Punct(punct_with_span('$', Alone, dollar_span)), TT::Ident(name)]),
                dollar_span,
            )),
            TT::Punct(punct_with_span('*', Alone, dollar_span)),
        ]);
    }

    fn add_arg(&mut self, dollar_span: Span, tt: TT, input: &mut IntoIter, out: &mut TokenStream) -> Result<()> {
        match tt {
            TT::Punct(p) if p.as_char() == ESCAPE_CHAR => out.extend([TT::Punct(p)]),
            TT::Punct(p) if p.as_char() == '\'' && p.spacing() == Joint => {
                let lt_name = expect_tt(
                    input.next(),
                    |tt| match tt {
                        TT::Ident(x) => Some(x),
                        _ => None,
                    },
                    "lifetime name",
                    p.span(),
                )?;
                let arg_span = p.span().join(lt_name.span()).unwrap_or(p.span());
                self.add_single_arg_def("lifetime", dollar_span, arg_span, out);
                self.args.extend([TT::Punct(p), TT::Ident(lt_name)]);
            },
            TT::Ident(x) => {
                self.add_single_arg_def("ident", dollar_span, x.span(), out);
                self.args.push(TT::Ident(x));
            },
            TT::Literal(x) => {
                self.add_single_arg_def("literal", dollar_span, x.span(), out);
                self.args.push(TT::Literal(x));
            },
            TT::Group(g) if g.delimiter() == Parenthesis => {
                let mut inner = g.stream().into_iter();
                if let Some(TT::Punct(p)) = inner.next()
                    && p.as_char() == '@'
                {
                    let kind = expect_tt(
                        inner.next(),
                        |tt| match tt {
                            TT::Ident(kind) => Some(kind),
                            _ => None,
                        },
                        "a macro fragment specifier",
                        p.span(),
                    )?;
                    self.add_parenthesized_arg_def(kind, dollar_span, g.span(), out);
                    self.args.push(TT::Group(group_with_span(Parenthesis, inner.collect(), g.span())))
                } else {
                    self.add_multi_arg_def(dollar_span, g.span(), out);
                    self.args.push(TT::Group(g));
                }
            },
            tt => return Err(make_error("unsupported escape", tt.span())),
        };
        Ok(())
    }

    fn add_sub_args(&mut self, args: Vec<TT>, out: &mut TokenStream) {
        self.add_multi_arg_def(Span::call_site(), Span::call_site(), out);
        self.args
            .extend([TT::Group(Group::new(Parenthesis, TokenStream::from_iter(args)))]);
    }
}

#[derive(Default)]
struct Expander {
    arm: Option<MacroArm>,
    expn: TokenStream,
}
impl Expander {
    fn for_arm(idx: usize) -> Self {
        Self {
            arm: Some(MacroArm {
                args_def: TokenStream::from_iter([TT::Literal(Literal::usize_unsuffixed(idx))]),
                args: Vec::new(),
            }),
            expn: TokenStream::new(),
        }
    }

    fn write_tt(&mut self, tt: TT, mac: &mut MacWriter) -> Result<()> {
        match tt {
            TT::Group(g) => {
                let outer = mem::take(&mut self.expn);
                self.expand(g.stream().into_iter(), mac)?;
                let inner = mem::replace(&mut self.expn, outer);
                self.expn
                    .extend([TT::Group(group_with_span(g.delimiter(), inner, g.span()))]);
            },
            tt => self.expn.extend([tt]),
        }
        Ok(())
    }

    fn expand(&mut self, input: IntoIter, mac: &mut MacWriter) -> Result<()> {
        let Some(mut input) = LookaheadIter::new(input) else {
            return Ok(());
        };
        while let Some(tt) = input.next() {
            if let TT::Punct(p) = &tt
                && p.as_char() == ESCAPE_CHAR
                && let Some(arm) = self.arm.as_mut()
            {
                arm.add_arg(p.span(), mem::replace(&mut input.tt, tt), &mut input.iter, &mut self.expn)?;
                if input.next().is_none() {
                    return Ok(());
                }
            } else if let TT::Punct(p) = &input.tt
                && p.as_char() == '!'
                && let TT::Ident(name) = &tt
                && name.to_string() == "inline"
            {
                let g = expect_tt(
                    input.iter.next(),
                    |tt| match tt {
                        TT::Group(g) => Some(g),
                        _ => None,
                    },
                    "macro arguments",
                    p.span(),
                )?;
                mac.insert(name.span(), p.span(), g, self)?;
                if input.next().is_none() {
                    return Ok(());
                }
            } else {
                self.write_tt(tt, mac)?;
            }
        }
        self.write_tt(input.tt, mac)
    }
}
