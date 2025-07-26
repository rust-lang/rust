use rustc_ast::ptr::P;
use rustc_ast::{self as ast, AsmMacro};
use rustc_span::{Span, Symbol, kw};

use super::{ExpKeywordPair, ForceCollect, IdentIsRaw, Trailing, UsePreAttrPos};
use crate::{PResult, Parser, errors, exp, token};

/// An argument to one of the `asm!` macros. The argument is syntactically valid, but is otherwise
/// not validated at all.
pub struct AsmArg {
    pub kind: AsmArgKind,
    pub attributes: AsmAttrVec,
    pub span: Span,
}

pub enum AsmArgKind {
    Template(P<ast::Expr>),
    Operand(Option<Symbol>, ast::InlineAsmOperand),
    Options(Vec<AsmOption>),
    ClobberAbi(Vec<(Symbol, Span)>),
}

pub struct AsmOption {
    pub symbol: Symbol,
    pub span: Span,
    // A bitset, with only the bit for this option's symbol set.
    pub options: ast::InlineAsmOptions,
    // Used when suggesting to remove an option.
    pub span_with_comma: Span,
}

/// A parsed list of attributes that is not attached to any item.
/// Used to check whether `asm!` arguments are configured out.
pub struct AsmAttrVec(pub ast::AttrVec);

impl AsmAttrVec {
    fn parse<'a>(p: &mut Parser<'a>) -> PResult<'a, Self> {
        let attrs = p.parse_outer_attributes()?;

        p.collect_tokens(None, attrs, ForceCollect::No, |_, attrs| {
            Ok((Self(attrs), Trailing::No, UsePreAttrPos::No))
        })
    }
}
impl ast::HasAttrs for AsmAttrVec {
    // Follows `ast::Expr`.
    const SUPPORTS_CUSTOM_INNER_ATTRS: bool = false;

    fn attrs(&self) -> &[rustc_ast::Attribute] {
        &self.0
    }

    fn visit_attrs(&mut self, f: impl FnOnce(&mut rustc_ast::AttrVec)) {
        f(&mut self.0)
    }
}

impl ast::HasTokens for AsmAttrVec {
    fn tokens(&self) -> Option<&rustc_ast::tokenstream::LazyAttrTokenStream> {
        None
    }

    fn tokens_mut(&mut self) -> Option<&mut Option<rustc_ast::tokenstream::LazyAttrTokenStream>> {
        None
    }
}

/// Used for better error messages when operand types are used that are not
/// supported by the current macro (e.g. `in` or `out` for `global_asm!`)
///
/// returns
///
/// - `Ok(true)` if the current token matches the keyword, and was expected
/// - `Ok(false)` if the current token does not match the keyword
/// - `Err(_)` if the current token matches the keyword, but was not expected
fn eat_operand_keyword<'a>(
    p: &mut Parser<'a>,
    exp: ExpKeywordPair,
    asm_macro: AsmMacro,
) -> PResult<'a, bool> {
    if matches!(asm_macro, AsmMacro::Asm) {
        Ok(p.eat_keyword(exp))
    } else {
        let span = p.token.span;
        if p.eat_keyword_noexpect(exp.kw) {
            // in gets printed as `r#in` otherwise
            let symbol = if exp.kw == kw::In { "in" } else { exp.kw.as_str() };
            Err(p.dcx().create_err(errors::AsmUnsupportedOperand {
                span,
                symbol,
                macro_name: asm_macro.macro_name(),
            }))
        } else {
            Ok(false)
        }
    }
}

fn parse_asm_operand<'a>(
    p: &mut Parser<'a>,
    asm_macro: AsmMacro,
) -> PResult<'a, Option<ast::InlineAsmOperand>> {
    let dcx = p.dcx();

    Ok(Some(if eat_operand_keyword(p, exp!(In), asm_macro)? {
        let reg = parse_reg(p)?;
        if p.eat_keyword(exp!(Underscore)) {
            let err = dcx.create_err(errors::AsmUnderscoreInput { span: p.token.span });
            return Err(err);
        }
        let expr = p.parse_expr()?;
        ast::InlineAsmOperand::In { reg, expr }
    } else if eat_operand_keyword(p, exp!(Out), asm_macro)? {
        let reg = parse_reg(p)?;
        let expr = if p.eat_keyword(exp!(Underscore)) { None } else { Some(p.parse_expr()?) };
        ast::InlineAsmOperand::Out { reg, expr, late: false }
    } else if eat_operand_keyword(p, exp!(Lateout), asm_macro)? {
        let reg = parse_reg(p)?;
        let expr = if p.eat_keyword(exp!(Underscore)) { None } else { Some(p.parse_expr()?) };
        ast::InlineAsmOperand::Out { reg, expr, late: true }
    } else if eat_operand_keyword(p, exp!(Inout), asm_macro)? {
        let reg = parse_reg(p)?;
        if p.eat_keyword(exp!(Underscore)) {
            let err = dcx.create_err(errors::AsmUnderscoreInput { span: p.token.span });
            return Err(err);
        }
        let expr = p.parse_expr()?;
        if p.eat(exp!(FatArrow)) {
            let out_expr =
                if p.eat_keyword(exp!(Underscore)) { None } else { Some(p.parse_expr()?) };
            ast::InlineAsmOperand::SplitInOut { reg, in_expr: expr, out_expr, late: false }
        } else {
            ast::InlineAsmOperand::InOut { reg, expr, late: false }
        }
    } else if eat_operand_keyword(p, exp!(Inlateout), asm_macro)? {
        let reg = parse_reg(p)?;
        if p.eat_keyword(exp!(Underscore)) {
            let err = dcx.create_err(errors::AsmUnderscoreInput { span: p.token.span });
            return Err(err);
        }
        let expr = p.parse_expr()?;
        if p.eat(exp!(FatArrow)) {
            let out_expr =
                if p.eat_keyword(exp!(Underscore)) { None } else { Some(p.parse_expr()?) };
            ast::InlineAsmOperand::SplitInOut { reg, in_expr: expr, out_expr, late: true }
        } else {
            ast::InlineAsmOperand::InOut { reg, expr, late: true }
        }
    } else if eat_operand_keyword(p, exp!(Label), asm_macro)? {
        let block = p.parse_block()?;
        ast::InlineAsmOperand::Label { block }
    } else if p.eat_keyword(exp!(Const)) {
        let anon_const = p.parse_expr_anon_const()?;
        ast::InlineAsmOperand::Const { anon_const }
    } else if p.eat_keyword(exp!(Sym)) {
        let expr = p.parse_expr()?;
        let ast::ExprKind::Path(qself, path) = &expr.kind else {
            let err = dcx.create_err(errors::AsmSymNoPath { span: expr.span });
            return Err(err);
        };
        let sym =
            ast::InlineAsmSym { id: ast::DUMMY_NODE_ID, qself: qself.clone(), path: path.clone() };
        ast::InlineAsmOperand::Sym { sym }
    } else {
        return Ok(None);
    }))
}

// Public for rustfmt.
pub fn parse_asm_args<'a>(
    p: &mut Parser<'a>,
    sp: Span,
    asm_macro: AsmMacro,
) -> PResult<'a, Vec<AsmArg>> {
    let dcx = p.dcx();

    if p.token == token::Eof {
        return Err(dcx.create_err(errors::AsmRequiresTemplate { span: sp }));
    }

    let mut args = Vec::new();

    let attributes = AsmAttrVec::parse(p)?;
    let first_template = p.parse_expr()?;
    args.push(AsmArg {
        span: first_template.span,
        kind: AsmArgKind::Template(first_template),
        attributes,
    });

    let mut allow_templates = true;

    while p.token != token::Eof {
        if !p.eat(exp!(Comma)) {
            if allow_templates {
                // After a template string, we always expect *only* a comma...
                return Err(dcx.create_err(errors::AsmExpectedComma { span: p.token.span }));
            } else {
                // ...after that delegate to `expect` to also include the other expected tokens.
                return Err(p.expect(exp!(Comma)).err().unwrap());
            }
        }

        // Accept trailing commas.
        if p.token == token::Eof {
            break;
        }

        let attributes = AsmAttrVec::parse(p)?;
        let span_start = p.token.span;

        // Parse `clobber_abi`.
        if p.eat_keyword(exp!(ClobberAbi)) {
            allow_templates = false;

            args.push(AsmArg {
                kind: AsmArgKind::ClobberAbi(parse_clobber_abi(p)?),
                span: span_start.to(p.prev_token.span),
                attributes,
            });

            continue;
        }

        // Parse `options`.
        if p.eat_keyword(exp!(Options)) {
            allow_templates = false;

            args.push(AsmArg {
                kind: AsmArgKind::Options(parse_options(p, asm_macro)?),
                span: span_start.to(p.prev_token.span),
                attributes,
            });

            continue;
        }

        // Parse operand names.
        let name = if p.token.is_ident() && p.look_ahead(1, |t| *t == token::Eq) {
            let (ident, _) = p.token.ident().unwrap();
            p.bump();
            p.expect(exp!(Eq))?;
            allow_templates = false;
            Some(ident.name)
        } else {
            None
        };

        if let Some(op) = parse_asm_operand(p, asm_macro)? {
            allow_templates = false;

            args.push(AsmArg {
                span: span_start.to(p.prev_token.span),
                kind: AsmArgKind::Operand(name, op),
                attributes,
            });
        } else if allow_templates {
            let template = p.parse_expr()?;
            // If it can't possibly expand to a string, provide diagnostics here to include other
            // things it could have been.
            match template.kind {
                ast::ExprKind::Lit(token_lit)
                    if matches!(
                        token_lit.kind,
                        token::LitKind::Str | token::LitKind::StrRaw(_)
                    ) => {}
                ast::ExprKind::MacCall(..) => {}
                _ => {
                    let err = dcx.create_err(errors::AsmExpectedOther {
                        span: template.span,
                        is_inline_asm: matches!(asm_macro, AsmMacro::Asm),
                    });
                    return Err(err);
                }
            }

            args.push(AsmArg {
                span: template.span,
                kind: AsmArgKind::Template(template),
                attributes,
            });
        } else {
            p.unexpected_any()?
        }
    }

    Ok(args)
}

fn parse_options<'a>(p: &mut Parser<'a>, asm_macro: AsmMacro) -> PResult<'a, Vec<AsmOption>> {
    p.expect(exp!(OpenParen))?;

    let mut asm_options = Vec::new();

    while !p.eat(exp!(CloseParen)) {
        const OPTIONS: [(ExpKeywordPair, ast::InlineAsmOptions); ast::InlineAsmOptions::COUNT] = [
            (exp!(Pure), ast::InlineAsmOptions::PURE),
            (exp!(Nomem), ast::InlineAsmOptions::NOMEM),
            (exp!(Readonly), ast::InlineAsmOptions::READONLY),
            (exp!(PreservesFlags), ast::InlineAsmOptions::PRESERVES_FLAGS),
            (exp!(Noreturn), ast::InlineAsmOptions::NORETURN),
            (exp!(Nostack), ast::InlineAsmOptions::NOSTACK),
            (exp!(MayUnwind), ast::InlineAsmOptions::MAY_UNWIND),
            (exp!(AttSyntax), ast::InlineAsmOptions::ATT_SYNTAX),
            (exp!(Raw), ast::InlineAsmOptions::RAW),
        ];

        'blk: {
            for (exp, options) in OPTIONS {
                // Gives a more accurate list of expected next tokens.
                let kw_matched = if asm_macro.is_supported_option(options) {
                    p.eat_keyword(exp)
                } else {
                    p.eat_keyword_noexpect(exp.kw)
                };

                if kw_matched {
                    let span = p.prev_token.span;
                    let span_with_comma =
                        if p.token == token::Comma { span.to(p.token.span) } else { span };

                    asm_options.push(AsmOption { symbol: exp.kw, span, options, span_with_comma });
                    break 'blk;
                }
            }

            return p.unexpected_any();
        }

        // Allow trailing commas.
        if p.eat(exp!(CloseParen)) {
            break;
        }
        p.expect(exp!(Comma))?;
    }

    Ok(asm_options)
}

fn parse_clobber_abi<'a>(p: &mut Parser<'a>) -> PResult<'a, Vec<(Symbol, Span)>> {
    p.expect(exp!(OpenParen))?;

    if p.eat(exp!(CloseParen)) {
        return Err(p.dcx().create_err(errors::NonABI { span: p.token.span }));
    }

    let mut new_abis = Vec::new();
    while !p.eat(exp!(CloseParen)) {
        match p.parse_str_lit() {
            Ok(str_lit) => {
                new_abis.push((str_lit.symbol_unescaped, str_lit.span));
            }
            Err(opt_lit) => {
                let span = opt_lit.map_or(p.token.span, |lit| lit.span);
                return Err(p.dcx().create_err(errors::AsmExpectedStringLiteral { span }));
            }
        };

        // Allow trailing commas
        if p.eat(exp!(CloseParen)) {
            break;
        }
        p.expect(exp!(Comma))?;
    }

    Ok(new_abis)
}

fn parse_reg<'a>(p: &mut Parser<'a>) -> PResult<'a, ast::InlineAsmRegOrRegClass> {
    p.expect(exp!(OpenParen))?;
    let result = match p.token.uninterpolate().kind {
        token::Ident(name, IdentIsRaw::No) => ast::InlineAsmRegOrRegClass::RegClass(name),
        token::Literal(token::Lit { kind: token::LitKind::Str, symbol, suffix: _ }) => {
            ast::InlineAsmRegOrRegClass::Reg(symbol)
        }
        _ => {
            return Err(p.dcx().create_err(errors::ExpectedRegisterClassOrExplicitRegister {
                span: p.token.span,
            }));
        }
    };
    p.bump();
    p.expect(exp!(CloseParen))?;
    Ok(result)
}
