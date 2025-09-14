use rustc_ast::tokenstream::TokenStream;
use rustc_ast::{AsmMacro, token};
use rustc_data_structures::fx::{FxHashMap, FxIndexMap};
use rustc_errors::PResult;
use rustc_expand::base::*;
use rustc_index::bit_set::GrowableBitSet;
use rustc_parse::parser::asm::*;
use rustc_session::lint;
use rustc_session::parse::feature_err;
use rustc_span::{ErrorGuaranteed, InnerSpan, Span, Symbol, sym};
use rustc_target::asm::InlineAsmArch;
use smallvec::smallvec;
use {rustc_ast as ast, rustc_parse_format as parse};

use crate::util::{ExprToSpannedString, expr_to_spanned_string};
use crate::{errors, fluent_generated as fluent};

/// Validated assembly arguments, ready for macro expansion.
struct ValidatedAsmArgs {
    pub templates: Vec<Box<ast::Expr>>,
    pub operands: Vec<(ast::InlineAsmOperand, Span)>,
    named_args: FxIndexMap<Symbol, usize>,
    reg_args: GrowableBitSet<usize>,
    pub clobber_abis: Vec<(Symbol, Span)>,
    options: ast::InlineAsmOptions,
    pub options_spans: Vec<Span>,
}

fn parse_args<'a>(
    ecx: &ExtCtxt<'a>,
    sp: Span,
    tts: TokenStream,
    asm_macro: AsmMacro,
) -> PResult<'a, ValidatedAsmArgs> {
    let args = parse_asm_args(&mut ecx.new_parser_from_tts(tts), sp, asm_macro)?;
    validate_asm_args(ecx, asm_macro, args)
}

fn validate_asm_args<'a>(
    ecx: &ExtCtxt<'a>,
    asm_macro: AsmMacro,
    args: Vec<AsmArg>,
) -> PResult<'a, ValidatedAsmArgs> {
    let dcx = ecx.dcx();

    let strip_unconfigured = rustc_expand::config::StripUnconfigured {
        sess: ecx.sess,
        features: Some(ecx.ecfg.features),
        config_tokens: false,
        lint_node_id: ecx.current_expansion.lint_node_id,
    };

    let mut validated = ValidatedAsmArgs {
        templates: vec![],
        operands: vec![],
        named_args: Default::default(),
        reg_args: Default::default(),
        clobber_abis: Vec::new(),
        options: ast::InlineAsmOptions::empty(),
        options_spans: vec![],
    };

    let mut allow_templates = true;

    for arg in args {
        for attr in arg.attributes.0.iter() {
            match attr.name() {
                Some(sym::cfg | sym::cfg_attr) => {
                    if !ecx.ecfg.features.asm_cfg() {
                        let span = attr.span();
                        feature_err(ecx.sess, sym::asm_cfg, span, fluent::builtin_macros_asm_cfg)
                            .emit();
                    }
                }
                _ => {
                    ecx.dcx().emit_err(errors::AsmAttributeNotSupported { span: attr.span() });
                }
            }
        }

        // Skip arguments that are configured out.
        if ecx.ecfg.features.asm_cfg() && strip_unconfigured.configure(arg.attributes).is_none() {
            continue;
        }

        match arg.kind {
            AsmArgKind::Template(template) => {
                // The error for the first template is delayed.
                if !allow_templates {
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
                }

                validated.templates.push(template);
            }
            AsmArgKind::Operand(name, op) => {
                allow_templates = false;

                let explicit_reg = matches!(op.reg(), Some(ast::InlineAsmRegOrRegClass::Reg(_)));
                let span = arg.span;
                let slot = validated.operands.len();
                validated.operands.push((op, span));

                // Validate the order of named, positional & explicit register operands and
                // clobber_abi/options. We do this at the end once we have the full span
                // of the argument available.

                if explicit_reg {
                    if name.is_some() {
                        dcx.emit_err(errors::AsmExplicitRegisterName { span });
                    }
                    validated.reg_args.insert(slot);
                } else if let Some(name) = name {
                    if let Some(&prev) = validated.named_args.get(&name) {
                        dcx.emit_err(errors::AsmDuplicateArg {
                            span,
                            name,
                            prev: validated.operands[prev].1,
                        });
                        continue;
                    }
                    validated.named_args.insert(name, slot);
                } else if !validated.named_args.is_empty() || !validated.reg_args.is_empty() {
                    let named =
                        validated.named_args.values().map(|p| validated.operands[*p].1).collect();
                    let explicit =
                        validated.reg_args.iter().map(|p| validated.operands[p].1).collect();

                    dcx.emit_err(errors::AsmPositionalAfter { span, named, explicit });
                }
            }
            AsmArgKind::Options(new_options) => {
                allow_templates = false;

                for asm_option in new_options {
                    let AsmOption { span, symbol, span_with_comma, options } = asm_option;

                    if !asm_macro.is_supported_option(options) {
                        // Tool-only output.
                        dcx.emit_err(errors::AsmUnsupportedOption {
                            span,
                            symbol,
                            span_with_comma,
                            macro_name: asm_macro.macro_name(),
                        });
                    } else if validated.options.contains(options) {
                        // Tool-only output.
                        dcx.emit_err(errors::AsmOptAlreadyprovided {
                            span,
                            symbol,
                            span_with_comma,
                        });
                    } else {
                        validated.options |= asm_option.options;
                    }
                }

                validated.options_spans.push(arg.span);
            }
            AsmArgKind::ClobberAbi(new_abis) => {
                allow_templates = false;

                match &new_abis[..] {
                    // This should have errored above during parsing.
                    [] => unreachable!(),
                    [(abi, _span)] => validated.clobber_abis.push((*abi, arg.span)),
                    _ => validated.clobber_abis.extend(new_abis),
                }
            }
        }
    }

    if validated.options.contains(ast::InlineAsmOptions::NOMEM)
        && validated.options.contains(ast::InlineAsmOptions::READONLY)
    {
        let spans = validated.options_spans.clone();
        dcx.emit_err(errors::AsmMutuallyExclusive { spans, opt1: "nomem", opt2: "readonly" });
    }
    if validated.options.contains(ast::InlineAsmOptions::PURE)
        && validated.options.contains(ast::InlineAsmOptions::NORETURN)
    {
        let spans = validated.options_spans.clone();
        dcx.emit_err(errors::AsmMutuallyExclusive { spans, opt1: "pure", opt2: "noreturn" });
    }
    if validated.options.contains(ast::InlineAsmOptions::PURE)
        && !validated
            .options
            .intersects(ast::InlineAsmOptions::NOMEM | ast::InlineAsmOptions::READONLY)
    {
        let spans = validated.options_spans.clone();
        dcx.emit_err(errors::AsmPureCombine { spans });
    }

    let mut have_real_output = false;
    let mut outputs_sp = vec![];
    let mut regclass_outputs = vec![];
    let mut labels_sp = vec![];
    for (op, op_sp) in &validated.operands {
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
    if validated.options.contains(ast::InlineAsmOptions::PURE) && !have_real_output {
        dcx.emit_err(errors::AsmPureNoOutput { spans: validated.options_spans.clone() });
    }
    if validated.options.contains(ast::InlineAsmOptions::NORETURN)
        && !outputs_sp.is_empty()
        && labels_sp.is_empty()
    {
        let err = dcx.create_err(errors::AsmNoReturn { outputs_sp });
        // Bail out now since this is likely to confuse MIR
        return Err(err);
    }
    if validated.options.contains(ast::InlineAsmOptions::MAY_UNWIND) && !labels_sp.is_empty() {
        dcx.emit_err(errors::AsmMayUnwind { labels_sp });
    }

    if !validated.clobber_abis.is_empty() {
        match asm_macro {
            AsmMacro::GlobalAsm | AsmMacro::NakedAsm => {
                let err = dcx.create_err(errors::AsmUnsupportedClobberAbi {
                    spans: validated.clobber_abis.iter().map(|(_, span)| *span).collect(),
                    macro_name: asm_macro.macro_name(),
                });

                // Bail out now since this is likely to confuse later stages
                return Err(err);
            }
            AsmMacro::Asm => {
                if !regclass_outputs.is_empty() {
                    dcx.emit_err(errors::AsmClobberNoReg {
                        spans: regclass_outputs,
                        clobbers: validated.clobber_abis.iter().map(|(_, span)| *span).collect(),
                    });
                }
            }
        }
    }

    Ok(validated)
}

fn expand_preparsed_asm(
    ecx: &mut ExtCtxt<'_>,
    asm_macro: AsmMacro,
    args: ValidatedAsmArgs,
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
                    errors::AvoidIntelSyntax,
                );
            }
            if template_str.contains(".att_syntax") {
                ecx.psess().buffer_lint(
                    lint::builtin::BAD_ASM_STYLE,
                    find_span(".att_syntax"),
                    ecx.current_expansion.lint_node_id,
                    errors::AvoidAttSyntax,
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
                Ok(inline_asm) => Box::new(ast::Expr {
                    id: ast::DUMMY_NODE_ID,
                    kind: ast::ExprKind::InlineAsm(Box::new(inline_asm)),
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
                Ok(inline_asm) => Box::new(ast::Expr {
                    id: ast::DUMMY_NODE_ID,
                    kind: ast::ExprKind::InlineAsm(Box::new(inline_asm)),
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
                Ok(inline_asm) => MacEager::items(smallvec![Box::new(ast::Item {
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
