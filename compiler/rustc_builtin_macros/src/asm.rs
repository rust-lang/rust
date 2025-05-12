use ast::token::IdentIsRaw;
use lint::BuiltinLintDiag;
use rustc_ast::ptr::P;
use rustc_ast::tokenstream::TokenStream;
use rustc_ast::{AsmMacro, token};
use rustc_data_structures::fx::{FxHashMap, FxIndexMap};
use rustc_errors::PResult;
use rustc_expand::base::*;
use rustc_index::bit_set::GrowableBitSet;
use rustc_parse::exp;
use rustc_parse::parser::{ExpKeywordPair, Parser};
use rustc_session::lint;
use rustc_span::{ErrorGuaranteed, InnerSpan, Span, Symbol, kw};
use rustc_target::asm::InlineAsmArch;
use smallvec::smallvec;
use {rustc_ast as ast, rustc_parse_format as parse};

use crate::errors;
use crate::util::{ExprToSpannedString, expr_to_spanned_string};

pub struct AsmArgs {
    pub templates: Vec<P<ast::Expr>>,
    pub operands: Vec<(ast::InlineAsmOperand, Span)>,
    named_args: FxIndexMap<Symbol, usize>,
    reg_args: GrowableBitSet<usize>,
    pub clobber_abis: Vec<(Symbol, Span)>,
    options: ast::InlineAsmOptions,
    pub options_spans: Vec<Span>,
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

fn parse_args<'a>(
    ecx: &ExtCtxt<'a>,
    sp: Span,
    tts: TokenStream,
    asm_macro: AsmMacro,
) -> PResult<'a, AsmArgs> {
    let mut p = ecx.new_parser_from_tts(tts);
    parse_asm_args(&mut p, sp, asm_macro)
}

// Primarily public for rustfmt consumption.
// Internal consumers should continue to leverage `expand_asm`/`expand__global_asm`
pub fn parse_asm_args<'a>(
    p: &mut Parser<'a>,
    sp: Span,
    asm_macro: AsmMacro,
) -> PResult<'a, AsmArgs> {
    let dcx = p.dcx();

    if p.token == token::Eof {
        return Err(dcx.create_err(errors::AsmRequiresTemplate { span: sp }));
    }

    let first_template = p.parse_expr()?;
    let mut args = AsmArgs {
        templates: vec![first_template],
        operands: vec![],
        named_args: Default::default(),
        reg_args: Default::default(),
        clobber_abis: Vec::new(),
        options: ast::InlineAsmOptions::empty(),
        options_spans: vec![],
    };

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
        if p.token == token::Eof {
            break;
        } // accept trailing commas

        // Parse clobber_abi
        if p.eat_keyword(exp!(ClobberAbi)) {
            parse_clobber_abi(p, &mut args)?;
            allow_templates = false;
            continue;
        }

        // Parse options
        if p.eat_keyword(exp!(Options)) {
            parse_options(p, &mut args, asm_macro)?;
            allow_templates = false;
            continue;
        }

        let span_start = p.token.span;

        // Parse operand names
        let name = if p.token.is_ident() && p.look_ahead(1, |t| *t == token::Eq) {
            let (ident, _) = p.token.ident().unwrap();
            p.bump();
            p.expect(exp!(Eq))?;
            allow_templates = false;
            Some(ident.name)
        } else {
            None
        };

        let mut explicit_reg = false;
        let op = if eat_operand_keyword(p, exp!(In), asm_macro)? {
            let reg = parse_reg(p, &mut explicit_reg)?;
            if p.eat_keyword(exp!(Underscore)) {
                let err = dcx.create_err(errors::AsmUnderscoreInput { span: p.token.span });
                return Err(err);
            }
            let expr = p.parse_expr()?;
            ast::InlineAsmOperand::In { reg, expr }
        } else if eat_operand_keyword(p, exp!(Out), asm_macro)? {
            let reg = parse_reg(p, &mut explicit_reg)?;
            let expr = if p.eat_keyword(exp!(Underscore)) { None } else { Some(p.parse_expr()?) };
            ast::InlineAsmOperand::Out { reg, expr, late: false }
        } else if eat_operand_keyword(p, exp!(Lateout), asm_macro)? {
            let reg = parse_reg(p, &mut explicit_reg)?;
            let expr = if p.eat_keyword(exp!(Underscore)) { None } else { Some(p.parse_expr()?) };
            ast::InlineAsmOperand::Out { reg, expr, late: true }
        } else if eat_operand_keyword(p, exp!(Inout), asm_macro)? {
            let reg = parse_reg(p, &mut explicit_reg)?;
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
            let reg = parse_reg(p, &mut explicit_reg)?;
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
            let sym = ast::InlineAsmSym {
                id: ast::DUMMY_NODE_ID,
                qself: qself.clone(),
                path: path.clone(),
            };
            ast::InlineAsmOperand::Sym { sym }
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
            args.templates.push(template);
            continue;
        } else {
            p.unexpected_any()?
        };

        allow_templates = false;
        let span = span_start.to(p.prev_token.span);
        let slot = args.operands.len();
        args.operands.push((op, span));

        // Validate the order of named, positional & explicit register operands and
        // clobber_abi/options. We do this at the end once we have the full span
        // of the argument available.
        if explicit_reg {
            if name.is_some() {
                dcx.emit_err(errors::AsmExplicitRegisterName { span });
            }
            args.reg_args.insert(slot);
        } else if let Some(name) = name {
            if let Some(&prev) = args.named_args.get(&name) {
                dcx.emit_err(errors::AsmDuplicateArg { span, name, prev: args.operands[prev].1 });
                continue;
            }
            args.named_args.insert(name, slot);
        } else if !args.named_args.is_empty() || !args.reg_args.is_empty() {
            let named = args.named_args.values().map(|p| args.operands[*p].1).collect();
            let explicit = args.reg_args.iter().map(|p| args.operands[p].1).collect();

            dcx.emit_err(errors::AsmPositionalAfter { span, named, explicit });
        }
    }

    if args.options.contains(ast::InlineAsmOptions::NOMEM)
        && args.options.contains(ast::InlineAsmOptions::READONLY)
    {
        let spans = args.options_spans.clone();
        dcx.emit_err(errors::AsmMutuallyExclusive { spans, opt1: "nomem", opt2: "readonly" });
    }
    if args.options.contains(ast::InlineAsmOptions::PURE)
        && args.options.contains(ast::InlineAsmOptions::NORETURN)
    {
        let spans = args.options_spans.clone();
        dcx.emit_err(errors::AsmMutuallyExclusive { spans, opt1: "pure", opt2: "noreturn" });
    }
    if args.options.contains(ast::InlineAsmOptions::PURE)
        && !args.options.intersects(ast::InlineAsmOptions::NOMEM | ast::InlineAsmOptions::READONLY)
    {
        let spans = args.options_spans.clone();
        dcx.emit_err(errors::AsmPureCombine { spans });
    }

    let mut have_real_output = false;
    let mut outputs_sp = vec![];
    let mut regclass_outputs = vec![];
    let mut labels_sp = vec![];
    for (op, op_sp) in &args.operands {
        match op {
            ast::InlineAsmOperand::Out { reg, expr, .. }
            | ast::InlineAsmOperand::SplitInOut { reg, out_expr: expr, .. } => {
                outputs_sp.push(*op_sp);
                have_real_output |= expr.is_some();
                if let ast::InlineAsmRegOrRegClass::RegClass(_) = reg {
                    regclass_outputs.push(*op_sp);
                }
            }
            ast::InlineAsmOperand::InOut { reg, .. } => {
                outputs_sp.push(*op_sp);
                have_real_output = true;
                if let ast::InlineAsmRegOrRegClass::RegClass(_) = reg {
                    regclass_outputs.push(*op_sp);
                }
            }
            ast::InlineAsmOperand::Label { .. } => {
                labels_sp.push(*op_sp);
            }
            _ => {}
        }
    }
    if args.options.contains(ast::InlineAsmOptions::PURE) && !have_real_output {
        dcx.emit_err(errors::AsmPureNoOutput { spans: args.options_spans.clone() });
    }
    if args.options.contains(ast::InlineAsmOptions::NORETURN)
        && !outputs_sp.is_empty()
        && labels_sp.is_empty()
    {
        let err = dcx.create_err(errors::AsmNoReturn { outputs_sp });
        // Bail out now since this is likely to confuse MIR
        return Err(err);
    }
    if args.options.contains(ast::InlineAsmOptions::MAY_UNWIND) && !labels_sp.is_empty() {
        dcx.emit_err(errors::AsmMayUnwind { labels_sp });
    }

    if !args.clobber_abis.is_empty() {
        match asm_macro {
            AsmMacro::GlobalAsm | AsmMacro::NakedAsm => {
                let err = dcx.create_err(errors::AsmUnsupportedClobberAbi {
                    spans: args.clobber_abis.iter().map(|(_, span)| *span).collect(),
                    macro_name: asm_macro.macro_name(),
                });

                // Bail out now since this is likely to confuse later stages
                return Err(err);
            }
            AsmMacro::Asm => {
                if !regclass_outputs.is_empty() {
                    dcx.emit_err(errors::AsmClobberNoReg {
                        spans: regclass_outputs,
                        clobbers: args.clobber_abis.iter().map(|(_, span)| *span).collect(),
                    });
                }
            }
        }
    }

    Ok(args)
}

/// Report a duplicate option error.
///
/// This function must be called immediately after the option token is parsed.
/// Otherwise, the suggestion will be incorrect.
fn err_duplicate_option(p: &Parser<'_>, symbol: Symbol, span: Span) {
    // Tool-only output
    let full_span = if p.token == token::Comma { span.to(p.token.span) } else { span };
    p.dcx().emit_err(errors::AsmOptAlreadyprovided { span, symbol, full_span });
}

/// Report an invalid option error.
///
/// This function must be called immediately after the option token is parsed.
/// Otherwise, the suggestion will be incorrect.
fn err_unsupported_option(p: &Parser<'_>, asm_macro: AsmMacro, symbol: Symbol, span: Span) {
    // Tool-only output
    let full_span = if p.token == token::Comma { span.to(p.token.span) } else { span };
    p.dcx().emit_err(errors::AsmUnsupportedOption {
        span,
        symbol,
        full_span,
        macro_name: asm_macro.macro_name(),
    });
}

/// Try to set the provided option in the provided `AsmArgs`.
/// If it is already set, report a duplicate option error.
///
/// This function must be called immediately after the option token is parsed.
/// Otherwise, the error will not point to the correct spot.
fn try_set_option<'a>(
    p: &Parser<'a>,
    args: &mut AsmArgs,
    asm_macro: AsmMacro,
    symbol: Symbol,
    option: ast::InlineAsmOptions,
) {
    if !asm_macro.is_supported_option(option) {
        err_unsupported_option(p, asm_macro, symbol, p.prev_token.span);
    } else if args.options.contains(option) {
        err_duplicate_option(p, symbol, p.prev_token.span);
    } else {
        args.options |= option;
    }
}

fn parse_options<'a>(
    p: &mut Parser<'a>,
    args: &mut AsmArgs,
    asm_macro: AsmMacro,
) -> PResult<'a, ()> {
    let span_start = p.prev_token.span;

    p.expect(exp!(OpenParen))?;

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
            for (exp, option) in OPTIONS {
                let kw_matched = if asm_macro.is_supported_option(option) {
                    p.eat_keyword(exp)
                } else {
                    p.eat_keyword_noexpect(exp.kw)
                };

                if kw_matched {
                    try_set_option(p, args, asm_macro, exp.kw, option);
                    break 'blk;
                }
            }

            return p.unexpected();
        }

        // Allow trailing commas
        if p.eat(exp!(CloseParen)) {
            break;
        }
        p.expect(exp!(Comma))?;
    }

    let new_span = span_start.to(p.prev_token.span);
    args.options_spans.push(new_span);

    Ok(())
}

fn parse_clobber_abi<'a>(p: &mut Parser<'a>, args: &mut AsmArgs) -> PResult<'a, ()> {
    let span_start = p.prev_token.span;

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

    let full_span = span_start.to(p.prev_token.span);

    match &new_abis[..] {
        // should have errored above during parsing
        [] => unreachable!(),
        [(abi, _span)] => args.clobber_abis.push((*abi, full_span)),
        abis => {
            for (abi, span) in abis {
                args.clobber_abis.push((*abi, *span));
            }
        }
    }

    Ok(())
}

fn parse_reg<'a>(
    p: &mut Parser<'a>,
    explicit_reg: &mut bool,
) -> PResult<'a, ast::InlineAsmRegOrRegClass> {
    p.expect(exp!(OpenParen))?;
    let result = match p.token.uninterpolate().kind {
        token::Ident(name, IdentIsRaw::No) => ast::InlineAsmRegOrRegClass::RegClass(name),
        token::Literal(token::Lit { kind: token::LitKind::Str, symbol, suffix: _ }) => {
            *explicit_reg = true;
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

fn expand_preparsed_asm(
    ecx: &mut ExtCtxt<'_>,
    asm_macro: AsmMacro,
    args: AsmArgs,
) -> ExpandResult<Result<ast::InlineAsm, ErrorGuaranteed>, ()> {
    let mut template = vec![];
    // Register operands are implicitly used since they are not allowed to be
    // referenced in the template string.
    let mut used = vec![false; args.operands.len()];
    for pos in args.reg_args.iter() {
        used[pos] = true;
    }
    let named_pos: FxHashMap<usize, Symbol> =
        args.named_args.iter().map(|(&sym, &idx)| (idx, sym)).collect();
    let mut line_spans = Vec::with_capacity(args.templates.len());
    let mut curarg = 0;

    let mut template_strs = Vec::with_capacity(args.templates.len());

    for (i, template_expr) in args.templates.into_iter().enumerate() {
        if i != 0 {
            template.push(ast::InlineAsmTemplatePiece::String("\n".into()));
        }

        let msg = "asm template must be a string literal";
        let template_sp = template_expr.span;
        let template_is_mac_call = matches!(template_expr.kind, ast::ExprKind::MacCall(_));
        let ExprToSpannedString {
            symbol: template_str,
            style: template_style,
            span: template_span,
            ..
        } = {
            let ExpandResult::Ready(mac) = expr_to_spanned_string(ecx, template_expr, msg) else {
                return ExpandResult::Retry(());
            };
            match mac {
                Ok(template_part) => template_part,
                Err(err) => {
                    return ExpandResult::Ready(Err(match err {
                        Ok((err, _)) => err.emit(),
                        Err(guar) => guar,
                    }));
                }
            }
        };

        let str_style = match template_style {
            ast::StrStyle::Cooked => None,
            ast::StrStyle::Raw(raw) => Some(raw as usize),
        };

        let template_snippet = ecx.source_map().span_to_snippet(template_sp).ok();
        template_strs.push((
            template_str,
            template_snippet.as_deref().map(Symbol::intern),
            template_sp,
        ));
        let template_str = template_str.as_str();

        if let Some(InlineAsmArch::X86 | InlineAsmArch::X86_64) = ecx.sess.asm_arch {
            let find_span = |needle: &str| -> Span {
                if let Some(snippet) = &template_snippet {
                    if let Some(pos) = snippet.find(needle) {
                        let end = pos
                            + snippet[pos..]
                                .find(|c| matches!(c, '\n' | ';' | '\\' | '"'))
                                .unwrap_or(snippet[pos..].len() - 1);
                        let inner = InnerSpan::new(pos, end);
                        return template_sp.from_inner(inner);
                    }
                }
                template_sp
            };

            if template_str.contains(".intel_syntax") {
                ecx.psess().buffer_lint(
                    lint::builtin::BAD_ASM_STYLE,
                    find_span(".intel_syntax"),
                    ecx.current_expansion.lint_node_id,
                    BuiltinLintDiag::AvoidUsingIntelSyntax,
                );
            }
            if template_str.contains(".att_syntax") {
                ecx.psess().buffer_lint(
                    lint::builtin::BAD_ASM_STYLE,
                    find_span(".att_syntax"),
                    ecx.current_expansion.lint_node_id,
                    BuiltinLintDiag::AvoidUsingAttSyntax,
                );
            }
        }

        // Don't treat raw asm as a format string.
        if args.options.contains(ast::InlineAsmOptions::RAW) {
            template.push(ast::InlineAsmTemplatePiece::String(template_str.to_string().into()));
            let template_num_lines = 1 + template_str.matches('\n').count();
            line_spans.extend(std::iter::repeat(template_sp).take(template_num_lines));
            continue;
        }

        let mut parser = parse::Parser::new(
            template_str,
            str_style,
            template_snippet,
            false,
            parse::ParseMode::InlineAsm,
        );
        parser.curarg = curarg;

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
            let err_sp = if template_is_mac_call {
                // If the template is a macro call we can't reliably point to the error's
                // span so just use the template's span as the error span (fixes #129503)
                template_span
            } else {
                template_span.from_inner(InnerSpan::new(err.span.start, err.span.end))
            };

            let msg = format!("invalid asm template string: {}", err.description);
            let mut e = ecx.dcx().struct_span_err(err_sp, msg);
            e.span_label(err_sp, err.label + " in asm template string");
            if let Some(note) = err.note {
                e.note(note);
            }
            if let Some((label, span)) = err.secondary_label {
                let err_sp = template_span.from_inner(InnerSpan::new(span.start, span.end));
                e.span_label(err_sp, label);
            }
            let guar = e.emit();
            return ExpandResult::Ready(Err(guar));
        }

        curarg = parser.curarg;

        let mut arg_spans = parser
            .arg_places
            .iter()
            .map(|span| template_span.from_inner(InnerSpan::new(span.start, span.end)));
        for piece in unverified_pieces {
            match piece {
                parse::Piece::Lit(s) => {
                    template.push(ast::InlineAsmTemplatePiece::String(s.to_string().into()))
                }
                parse::Piece::NextArgument(arg) => {
                    let span = arg_spans.next().unwrap_or(template_sp);

                    let operand_idx = match arg.position {
                        parse::ArgumentIs(idx) | parse::ArgumentImplicitlyIs(idx) => {
                            if idx >= args.operands.len()
                                || named_pos.contains_key(&idx)
                                || args.reg_args.contains(idx)
                            {
                                let msg = format!("invalid reference to argument at index {idx}");
                                let mut err = ecx.dcx().struct_span_err(span, msg);
                                err.span_label(span, "from here");

                                let positional_args = args.operands.len()
                                    - args.named_args.len()
                                    - args.reg_args.len();
                                let positional = if positional_args != args.operands.len() {
                                    "positional "
                                } else {
                                    ""
                                };
                                let msg = match positional_args {
                                    0 => format!("no {positional}arguments were given"),
                                    1 => format!("there is 1 {positional}argument"),
                                    x => format!("there are {x} {positional}arguments"),
                                };
                                err.note(msg);

                                if named_pos.contains_key(&idx) {
                                    err.span_label(args.operands[idx].1, "named argument");
                                    err.span_note(
                                        args.operands[idx].1,
                                        "named arguments cannot be referenced by position",
                                    );
                                } else if args.reg_args.contains(idx) {
                                    err.span_label(
                                        args.operands[idx].1,
                                        "explicit register argument",
                                    );
                                    err.span_note(
                                        args.operands[idx].1,
                                        "explicit register arguments cannot be used in the asm template",
                                    );
                                    err.span_help(
                                        args.operands[idx].1,
                                        "use the register name directly in the assembly code",
                                    );
                                }
                                err.emit();
                                None
                            } else {
                                Some(idx)
                            }
                        }
                        parse::ArgumentNamed(name) => {
                            match args.named_args.get(&Symbol::intern(name)) {
                                Some(&idx) => Some(idx),
                                None => {
                                    let span = arg.position_span;
                                    ecx.dcx()
                                        .create_err(errors::AsmNoMatchedArgumentName {
                                            name: name.to_owned(),
                                            span: template_span
                                                .from_inner(InnerSpan::new(span.start, span.end)),
                                        })
                                        .emit();
                                    None
                                }
                            }
                        }
                    };

                    let mut chars = arg.format.ty.chars();
                    let mut modifier = chars.next();
                    if chars.next().is_some() {
                        let span = arg
                            .format
                            .ty_span
                            .map(|sp| template_sp.from_inner(InnerSpan::new(sp.start, sp.end)))
                            .unwrap_or(template_sp);
                        ecx.dcx().emit_err(errors::AsmModifierInvalid { span });
                        modifier = None;
                    }

                    if let Some(operand_idx) = operand_idx {
                        used[operand_idx] = true;
                        template.push(ast::InlineAsmTemplatePiece::Placeholder {
                            operand_idx,
                            modifier,
                            span,
                        });
                    }
                }
            }
        }

        if parser.line_spans.is_empty() {
            let template_num_lines = 1 + template_str.matches('\n').count();
            line_spans.extend(std::iter::repeat(template_sp).take(template_num_lines));
        } else {
            line_spans.extend(
                parser
                    .line_spans
                    .iter()
                    .map(|span| template_span.from_inner(InnerSpan::new(span.start, span.end))),
            );
        };
    }

    let mut unused_operands = vec![];
    let mut help_str = String::new();
    for (idx, used) in used.into_iter().enumerate() {
        if !used {
            let msg = if let Some(sym) = named_pos.get(&idx) {
                help_str.push_str(&format!(" {{{}}}", sym));
                "named argument never used"
            } else {
                help_str.push_str(&format!(" {{{}}}", idx));
                "argument never used"
            };
            unused_operands.push((args.operands[idx].1, msg));
        }
    }
    match unused_operands[..] {
        [] => {}
        [(sp, msg)] => {
            ecx.dcx()
                .struct_span_err(sp, msg)
                .with_span_label(sp, msg)
                .with_help(format!(
                    "if this argument is intentionally unused, \
                     consider using it in an asm comment: `\"/*{help_str} */\"`"
                ))
                .emit();
        }
        _ => {
            let mut err = ecx.dcx().struct_span_err(
                unused_operands.iter().map(|&(sp, _)| sp).collect::<Vec<Span>>(),
                "multiple unused asm arguments",
            );
            for (sp, msg) in unused_operands {
                err.span_label(sp, msg);
            }
            err.help(format!(
                "if these arguments are intentionally unused, \
                 consider using them in an asm comment: `\"/*{help_str} */\"`"
            ));
            err.emit();
        }
    }

    ExpandResult::Ready(Ok(ast::InlineAsm {
        asm_macro,
        template,
        template_strs: template_strs.into_boxed_slice(),
        operands: args.operands,
        clobber_abis: args.clobber_abis,
        options: args.options,
        line_spans,
    }))
}

pub(super) fn expand_asm<'cx>(
    ecx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'cx> {
    ExpandResult::Ready(match parse_args(ecx, sp, tts, AsmMacro::Asm) {
        Ok(args) => {
            let ExpandResult::Ready(mac) = expand_preparsed_asm(ecx, AsmMacro::Asm, args) else {
                return ExpandResult::Retry(());
            };
            let expr = match mac {
                Ok(inline_asm) => P(ast::Expr {
                    id: ast::DUMMY_NODE_ID,
                    kind: ast::ExprKind::InlineAsm(P(inline_asm)),
                    span: sp,
                    attrs: ast::AttrVec::new(),
                    tokens: None,
                }),
                Err(guar) => DummyResult::raw_expr(sp, Some(guar)),
            };
            MacEager::expr(expr)
        }
        Err(err) => {
            let guar = err.emit();
            DummyResult::any(sp, guar)
        }
    })
}

pub(super) fn expand_naked_asm<'cx>(
    ecx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'cx> {
    ExpandResult::Ready(match parse_args(ecx, sp, tts, AsmMacro::NakedAsm) {
        Ok(args) => {
            let ExpandResult::Ready(mac) = expand_preparsed_asm(ecx, AsmMacro::NakedAsm, args)
            else {
                return ExpandResult::Retry(());
            };
            let expr = match mac {
                Ok(inline_asm) => P(ast::Expr {
                    id: ast::DUMMY_NODE_ID,
                    kind: ast::ExprKind::InlineAsm(P(inline_asm)),
                    span: sp,
                    attrs: ast::AttrVec::new(),
                    tokens: None,
                }),
                Err(guar) => DummyResult::raw_expr(sp, Some(guar)),
            };
            MacEager::expr(expr)
        }
        Err(err) => {
            let guar = err.emit();
            DummyResult::any(sp, guar)
        }
    })
}

pub(super) fn expand_global_asm<'cx>(
    ecx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'cx> {
    ExpandResult::Ready(match parse_args(ecx, sp, tts, AsmMacro::GlobalAsm) {
        Ok(args) => {
            let ExpandResult::Ready(mac) = expand_preparsed_asm(ecx, AsmMacro::GlobalAsm, args)
            else {
                return ExpandResult::Retry(());
            };
            match mac {
                Ok(inline_asm) => MacEager::items(smallvec![P(ast::Item {
                    attrs: ast::AttrVec::new(),
                    id: ast::DUMMY_NODE_ID,
                    kind: ast::ItemKind::GlobalAsm(Box::new(inline_asm)),
                    vis: ast::Visibility {
                        span: sp.shrink_to_lo(),
                        kind: ast::VisibilityKind::Inherited,
                        tokens: None,
                    },
                    span: sp,
                    tokens: None,
                })]),
                Err(guar) => DummyResult::any(sp, guar),
            }
        }
        Err(err) => {
            let guar = err.emit();
            DummyResult::any(sp, guar)
        }
    })
}
