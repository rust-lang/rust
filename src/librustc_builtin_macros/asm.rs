use fmt_macros as parse;

use rustc_ast::ast;
use rustc_ast::ptr::P;
use rustc_ast::token;
use rustc_ast::tokenstream::TokenStream;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::{Applicability, DiagnosticBuilder};
use rustc_expand::base::{self, *};
use rustc_parse::parser::Parser;
use rustc_span::symbol::{kw, sym, Symbol};
use rustc_span::{InnerSpan, Span};

struct AsmArgs {
    template: P<ast::Expr>,
    operands: Vec<(ast::InlineAsmOperand, Span)>,
    named_args: FxHashMap<Symbol, usize>,
    reg_args: FxHashSet<usize>,
    options: ast::InlineAsmOptions,
    options_span: Option<Span>,
}

fn parse_args<'a>(
    ecx: &mut ExtCtxt<'a>,
    sp: Span,
    tts: TokenStream,
) -> Result<AsmArgs, DiagnosticBuilder<'a>> {
    let mut p = ecx.new_parser_from_tts(tts);

    if p.token == token::Eof {
        return Err(ecx.struct_span_err(sp, "requires at least a template string argument"));
    }

    // Detect use of the legacy llvm_asm! syntax (which used to be called asm!)
    if p.look_ahead(1, |t| *t == token::Colon || *t == token::ModSep) {
        let mut err = ecx.struct_span_err(sp, "legacy asm! syntax is no longer supported");

        // Find the span of the "asm!" so that we can offer an automatic suggestion
        let asm_span = sp.from_inner(InnerSpan::new(0, 4));
        if let Ok(s) = ecx.source_map().span_to_snippet(asm_span) {
            if s == "asm!" {
                err.span_suggestion(
                    asm_span,
                    "replace with",
                    "llvm_asm!".into(),
                    Applicability::MachineApplicable,
                );
            }
        }
        return Err(err);
    }

    let template = p.parse_expr()?;
    let mut args = AsmArgs {
        template,
        operands: vec![],
        named_args: FxHashMap::default(),
        reg_args: FxHashSet::default(),
        options: ast::InlineAsmOptions::empty(),
        options_span: None,
    };

    let mut first = true;
    while p.token != token::Eof {
        if !p.eat(&token::Comma) {
            if first {
                // After `asm!(""` we always expect *only* a comma...
                let mut err = ecx.struct_span_err(p.token.span, "expected token: `,`");
                err.span_label(p.token.span, "expected `,`");
                p.maybe_annotate_with_ascription(&mut err, false);
                return Err(err);
            } else {
                // ...after that delegate to `expect` to also include the other expected tokens.
                return Err(p.expect(&token::Comma).err().unwrap());
            }
        }
        first = false;
        if p.token == token::Eof {
            break;
        } // accept trailing commas

        // Parse options
        if p.eat(&token::Ident(sym::options, false)) {
            parse_options(&mut p, &mut args)?;
            continue;
        }

        let span_start = p.token.span;

        // Parse operand names
        let name = if p.token.is_ident() && p.look_ahead(1, |t| *t == token::Eq) {
            let (ident, _) = p.token.ident().unwrap();
            p.bump();
            p.expect(&token::Eq)?;
            Some(ident.name)
        } else {
            None
        };

        let mut explicit_reg = false;
        let op = if p.eat(&token::Ident(kw::In, false)) {
            let reg = parse_reg(&mut p, &mut explicit_reg)?;
            let expr = p.parse_expr()?;
            ast::InlineAsmOperand::In { reg, expr }
        } else if p.eat(&token::Ident(sym::out, false)) {
            let reg = parse_reg(&mut p, &mut explicit_reg)?;
            let expr = if p.eat_keyword(kw::Underscore) { None } else { Some(p.parse_expr()?) };
            ast::InlineAsmOperand::Out { reg, expr, late: false }
        } else if p.eat(&token::Ident(sym::lateout, false)) {
            let reg = parse_reg(&mut p, &mut explicit_reg)?;
            let expr = if p.eat_keyword(kw::Underscore) { None } else { Some(p.parse_expr()?) };
            ast::InlineAsmOperand::Out { reg, expr, late: true }
        } else if p.eat(&token::Ident(sym::inout, false)) {
            let reg = parse_reg(&mut p, &mut explicit_reg)?;
            let expr = p.parse_expr()?;
            if p.eat(&token::FatArrow) {
                let out_expr =
                    if p.eat_keyword(kw::Underscore) { None } else { Some(p.parse_expr()?) };
                ast::InlineAsmOperand::SplitInOut { reg, in_expr: expr, out_expr, late: false }
            } else {
                ast::InlineAsmOperand::InOut { reg, expr, late: false }
            }
        } else if p.eat(&token::Ident(sym::inlateout, false)) {
            let reg = parse_reg(&mut p, &mut explicit_reg)?;
            let expr = p.parse_expr()?;
            if p.eat(&token::FatArrow) {
                let out_expr =
                    if p.eat_keyword(kw::Underscore) { None } else { Some(p.parse_expr()?) };
                ast::InlineAsmOperand::SplitInOut { reg, in_expr: expr, out_expr, late: true }
            } else {
                ast::InlineAsmOperand::InOut { reg, expr, late: true }
            }
        } else if p.eat(&token::Ident(kw::Const, false)) {
            let expr = p.parse_expr()?;
            ast::InlineAsmOperand::Const { expr }
        } else {
            p.expect(&token::Ident(sym::sym, false))?;
            let expr = p.parse_expr()?;
            match expr.kind {
                ast::ExprKind::Path(..) => {}
                _ => {
                    let err = ecx
                        .struct_span_err(expr.span, "argument to `sym` must be a path expression");
                    return Err(err);
                }
            }
            ast::InlineAsmOperand::Sym { expr }
        };

        let span = span_start.to(p.prev_token.span);
        let slot = args.operands.len();
        args.operands.push((op, span));

        // Validate the order of named, positional & explicit register operands and options. We do
        // this at the end once we have the full span of the argument available.
        if let Some(options_span) = args.options_span {
            ecx.struct_span_err(span, "arguments are not allowed after options")
                .span_label(options_span, "previous options")
                .span_label(span, "argument")
                .emit();
        }
        if explicit_reg {
            if name.is_some() {
                ecx.struct_span_err(span, "explicit register arguments cannot have names").emit();
            }
            args.reg_args.insert(slot);
        } else if let Some(name) = name {
            if let Some(&prev) = args.named_args.get(&name) {
                ecx.struct_span_err(span, &format!("duplicate argument named `{}`", name))
                    .span_label(args.operands[prev].1, "previously here")
                    .span_label(span, "duplicate argument")
                    .emit();
                continue;
            }
            if !args.reg_args.is_empty() {
                let mut err = ecx.struct_span_err(
                    span,
                    "named arguments cannot follow explicit register arguments",
                );
                err.span_label(span, "named argument");
                for pos in &args.reg_args {
                    err.span_label(args.operands[*pos].1, "explicit register argument");
                }
                err.emit();
            }
            args.named_args.insert(name, slot);
        } else {
            if !args.named_args.is_empty() || !args.reg_args.is_empty() {
                let mut err = ecx.struct_span_err(
                    span,
                    "positional arguments cannot follow named arguments \
                     or explicit register arguments",
                );
                err.span_label(span, "positional argument");
                for pos in args.named_args.values() {
                    err.span_label(args.operands[*pos].1, "named argument");
                }
                for pos in &args.reg_args {
                    err.span_label(args.operands[*pos].1, "explicit register argument");
                }
                err.emit();
            }
        }
    }

    if args.options.contains(ast::InlineAsmOptions::NOMEM)
        && args.options.contains(ast::InlineAsmOptions::READONLY)
    {
        let span = args.options_span.unwrap();
        ecx.struct_span_err(span, "the `nomem` and `readonly` options are mutually exclusive")
            .emit();
    }
    if args.options.contains(ast::InlineAsmOptions::PURE)
        && args.options.contains(ast::InlineAsmOptions::NORETURN)
    {
        let span = args.options_span.unwrap();
        ecx.struct_span_err(span, "the `pure` and `noreturn` options are mutually exclusive")
            .emit();
    }
    if args.options.contains(ast::InlineAsmOptions::PURE)
        && !args.options.intersects(ast::InlineAsmOptions::NOMEM | ast::InlineAsmOptions::READONLY)
    {
        let span = args.options_span.unwrap();
        ecx.struct_span_err(
            span,
            "the `pure` option must be combined with either `nomem` or `readonly`",
        )
        .emit();
    }

    let mut have_real_output = false;
    let mut outputs_sp = vec![];
    for (op, op_sp) in &args.operands {
        match op {
            ast::InlineAsmOperand::Out { expr, .. }
            | ast::InlineAsmOperand::SplitInOut { out_expr: expr, .. } => {
                outputs_sp.push(*op_sp);
                have_real_output |= expr.is_some();
            }
            ast::InlineAsmOperand::InOut { .. } => {
                outputs_sp.push(*op_sp);
                have_real_output = true;
            }
            _ => {}
        }
    }
    if args.options.contains(ast::InlineAsmOptions::PURE) && !have_real_output {
        ecx.struct_span_err(
            args.options_span.unwrap(),
            "asm with `pure` option must have at least one output",
        )
        .emit();
    }
    if args.options.contains(ast::InlineAsmOptions::NORETURN) && !outputs_sp.is_empty() {
        let err = ecx
            .struct_span_err(outputs_sp, "asm outputs are not allowed with the `noreturn` option");

        // Bail out now since this is likely to confuse MIR
        return Err(err);
    }

    Ok(args)
}

fn parse_options<'a>(p: &mut Parser<'a>, args: &mut AsmArgs) -> Result<(), DiagnosticBuilder<'a>> {
    let span_start = p.prev_token.span;

    p.expect(&token::OpenDelim(token::DelimToken::Paren))?;

    while !p.eat(&token::CloseDelim(token::DelimToken::Paren)) {
        if p.eat(&token::Ident(sym::pure, false)) {
            args.options |= ast::InlineAsmOptions::PURE;
        } else if p.eat(&token::Ident(sym::nomem, false)) {
            args.options |= ast::InlineAsmOptions::NOMEM;
        } else if p.eat(&token::Ident(sym::readonly, false)) {
            args.options |= ast::InlineAsmOptions::READONLY;
        } else if p.eat(&token::Ident(sym::preserves_flags, false)) {
            args.options |= ast::InlineAsmOptions::PRESERVES_FLAGS;
        } else if p.eat(&token::Ident(sym::noreturn, false)) {
            args.options |= ast::InlineAsmOptions::NORETURN;
        } else if p.eat(&token::Ident(sym::nostack, false)) {
            args.options |= ast::InlineAsmOptions::NOSTACK;
        } else {
            p.expect(&token::Ident(sym::att_syntax, false))?;
            args.options |= ast::InlineAsmOptions::ATT_SYNTAX;
        }

        // Allow trailing commas
        if p.eat(&token::CloseDelim(token::DelimToken::Paren)) {
            break;
        }
        p.expect(&token::Comma)?;
    }

    let new_span = span_start.to(p.prev_token.span);
    if let Some(options_span) = args.options_span {
        p.struct_span_err(new_span, "asm options cannot be specified multiple times")
            .span_label(options_span, "previously here")
            .span_label(new_span, "duplicate options")
            .emit();
    } else {
        args.options_span = Some(new_span);
    }

    Ok(())
}

fn parse_reg<'a>(
    p: &mut Parser<'a>,
    explicit_reg: &mut bool,
) -> Result<ast::InlineAsmRegOrRegClass, DiagnosticBuilder<'a>> {
    p.expect(&token::OpenDelim(token::DelimToken::Paren))?;
    let result = match p.token.kind {
        token::Ident(name, false) => ast::InlineAsmRegOrRegClass::RegClass(name),
        token::Literal(token::Lit { kind: token::LitKind::Str, symbol, suffix: _ }) => {
            *explicit_reg = true;
            ast::InlineAsmRegOrRegClass::Reg(symbol)
        }
        _ => {
            return Err(
                p.struct_span_err(p.token.span, "expected register class or explicit register")
            );
        }
    };
    p.bump();
    p.expect(&token::CloseDelim(token::DelimToken::Paren))?;
    Ok(result)
}

fn expand_preparsed_asm(ecx: &mut ExtCtxt<'_>, sp: Span, args: AsmArgs) -> P<ast::Expr> {
    let msg = "asm template must be a string literal";
    let template_sp = args.template.span;
    let (template_str, template_style, template_span) =
        match expr_to_spanned_string(ecx, args.template, msg) {
            Ok(template) => template,
            Err(err) => {
                if let Some(mut err) = err {
                    err.emit();
                }
                return DummyResult::raw_expr(sp, true);
            }
        };

    let str_style = match template_style {
        ast::StrStyle::Cooked => None,
        ast::StrStyle::Raw(raw) => Some(raw as usize),
    };

    let template_str = &template_str.as_str();
    let template_snippet = ecx.source_map().span_to_snippet(template_sp).ok();
    let mut parser = parse::Parser::new(
        template_str,
        str_style,
        template_snippet,
        false,
        parse::ParseMode::InlineAsm,
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
        let err_sp = template_span.from_inner(err.span);
        let mut e = ecx
            .struct_span_err(err_sp, &format!("invalid asm template string: {}", err.description));
        e.span_label(err_sp, err.label + " in asm template string");
        if let Some(note) = err.note {
            e.note(&note);
        }
        if let Some((label, span)) = err.secondary_label {
            let err_sp = template_span.from_inner(span);
            e.span_label(err_sp, label);
        }
        e.emit();
        return DummyResult::raw_expr(sp, true);
    }

    // Register operands are implicitly used since they are not allowed to be
    // referenced in the template string.
    let mut used = vec![false; args.operands.len()];
    for pos in &args.reg_args {
        used[*pos] = true;
    }

    let named_pos: FxHashSet<usize> = args.named_args.values().cloned().collect();
    let mut arg_spans = parser.arg_places.iter().map(|span| template_span.from_inner(*span));
    let mut template = vec![];
    for piece in unverified_pieces {
        match piece {
            parse::Piece::String(s) => {
                template.push(ast::InlineAsmTemplatePiece::String(s.to_string()))
            }
            parse::Piece::NextArgument(arg) => {
                let span = arg_spans.next().unwrap_or(template_sp);

                let operand_idx = match arg.position {
                    parse::ArgumentIs(idx) | parse::ArgumentImplicitlyIs(idx) => {
                        if idx >= args.operands.len()
                            || named_pos.contains(&idx)
                            || args.reg_args.contains(&idx)
                        {
                            let msg = format!("invalid reference to argument at index {}", idx);
                            let mut err = ecx.struct_span_err(span, &msg);
                            err.span_label(span, "from here");

                            let positional_args =
                                args.operands.len() - args.named_args.len() - args.reg_args.len();
                            let positional = if positional_args != args.operands.len() {
                                "positional "
                            } else {
                                ""
                            };
                            let msg = match positional_args {
                                0 => format!("no {}arguments were given", positional),
                                1 => format!("there is 1 {}argument", positional),
                                x => format!("there are {} {}arguments", x, positional),
                            };
                            err.note(&msg);

                            if named_pos.contains(&idx) {
                                err.span_label(args.operands[idx].1, "named argument");
                                err.span_note(
                                    args.operands[idx].1,
                                    "named arguments cannot be referenced by position",
                                );
                            } else if args.reg_args.contains(&idx) {
                                err.span_label(args.operands[idx].1, "explicit register argument");
                                err.span_note(
                                    args.operands[idx].1,
                                    "explicit register arguments cannot be used in the asm template",
                                );
                            }
                            err.emit();
                            None
                        } else {
                            Some(idx)
                        }
                    }
                    parse::ArgumentNamed(name) => match args.named_args.get(&name) {
                        Some(&idx) => Some(idx),
                        None => {
                            let msg = format!("there is no argument named `{}`", name);
                            ecx.struct_span_err(span, &msg[..]).emit();
                            None
                        }
                    },
                };

                let mut chars = arg.format.ty.chars();
                let mut modifier = chars.next();
                if !chars.next().is_none() {
                    let span = arg
                        .format
                        .ty_span
                        .map(|sp| template_sp.from_inner(sp))
                        .unwrap_or(template_sp);
                    ecx.struct_span_err(span, "asm template modifier must be a single character")
                        .emit();
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

    let operands = args.operands;
    let unused_operands: Vec<_> = used
        .into_iter()
        .enumerate()
        .filter(|&(_, used)| !used)
        .map(|(idx, _)| {
            if named_pos.contains(&idx) {
                // named argument
                (operands[idx].1, "named argument never used")
            } else {
                // positional argument
                (operands[idx].1, "argument never used")
            }
        })
        .collect();
    match unused_operands.len() {
        0 => {}
        1 => {
            let (sp, msg) = unused_operands.into_iter().next().unwrap();
            let mut err = ecx.struct_span_err(sp, msg);
            err.span_label(sp, msg);
            err.emit();
        }
        _ => {
            let mut err = ecx.struct_span_err(
                unused_operands.iter().map(|&(sp, _)| sp).collect::<Vec<Span>>(),
                "multiple unused asm arguments",
            );
            for (sp, msg) in unused_operands {
                err.span_label(sp, msg);
            }
            err.emit();
        }
    }

    let line_spans = if parser.line_spans.is_empty() {
        vec![template_sp]
    } else {
        parser.line_spans.iter().map(|span| template_span.from_inner(*span)).collect()
    };

    let inline_asm = ast::InlineAsm { template, operands, options: args.options, line_spans };
    P(ast::Expr {
        id: ast::DUMMY_NODE_ID,
        kind: ast::ExprKind::InlineAsm(P(inline_asm)),
        span: sp,
        attrs: ast::AttrVec::new(),
        tokens: None,
    })
}

pub fn expand_asm<'cx>(
    ecx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> Box<dyn base::MacResult + 'cx> {
    match parse_args(ecx, sp, tts) {
        Ok(args) => MacEager::expr(expand_preparsed_asm(ecx, sp, args)),
        Err(mut err) => {
            err.emit();
            DummyResult::any(sp)
        }
    }
}
