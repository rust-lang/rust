//! Lowering of inline assembly.
use hir_expand::name::Name;
use intern::Symbol;
use rustc_hash::{FxHashMap, FxHashSet};
use syntax::{
    AstNode, AstPtr, AstToken, T,
    ast::{self, HasName, IsString},
};
use tt::TextRange;

use crate::{
    expr_store::lower::{ExprCollector, FxIndexSet},
    hir::{AsmOperand, AsmOptions, Expr, ExprId, InlineAsm, InlineAsmKind, InlineAsmRegOrRegClass},
};

impl ExprCollector<'_> {
    pub(super) fn lower_inline_asm(
        &mut self,
        asm: ast::AsmExpr,
        syntax_ptr: AstPtr<ast::Expr>,
    ) -> ExprId {
        let mut clobber_abis = FxIndexSet::default();
        let mut operands = vec![];
        let mut options = AsmOptions::empty();

        let mut named_pos: FxHashMap<usize, Symbol> = Default::default();
        let mut named_args: FxHashMap<Symbol, usize> = Default::default();
        let mut reg_args: FxHashSet<usize> = Default::default();
        for piece in asm.asm_pieces() {
            let slot = operands.len();
            let mut lower_reg = |reg: Option<ast::AsmRegSpec>| {
                let reg = reg?;
                if let Some(string) = reg.string_token() {
                    reg_args.insert(slot);
                    Some(InlineAsmRegOrRegClass::Reg(Symbol::intern(string.text())))
                } else {
                    reg.name_ref().map(|name_ref| {
                        InlineAsmRegOrRegClass::RegClass(Symbol::intern(&name_ref.text()))
                    })
                }
            };

            let op = match piece {
                ast::AsmPiece::AsmClobberAbi(clobber_abi) => {
                    if let Some(abi_name) = clobber_abi.string_token() {
                        clobber_abis.insert(Symbol::intern(abi_name.text()));
                    }
                    continue;
                }
                ast::AsmPiece::AsmOptions(opt) => {
                    opt.asm_options().for_each(|opt| {
                        options |= match opt.syntax().first_token().map_or(T![$], |it| it.kind()) {
                            T![att_syntax] => AsmOptions::ATT_SYNTAX,
                            T![may_unwind] => AsmOptions::MAY_UNWIND,
                            T![nomem] => AsmOptions::NOMEM,
                            T![noreturn] => AsmOptions::NORETURN,
                            T![nostack] => AsmOptions::NOSTACK,
                            T![preserves_flags] => AsmOptions::PRESERVES_FLAGS,
                            T![pure] => AsmOptions::PURE,
                            T![raw] => AsmOptions::RAW,
                            T![readonly] => AsmOptions::READONLY,
                            _ => return,
                        }
                    });
                    continue;
                }
                ast::AsmPiece::AsmOperandNamed(op) => {
                    let name = op.name().map(|name| Symbol::intern(&name.text()));
                    if let Some(name) = &name {
                        named_args.insert(name.clone(), slot);
                        named_pos.insert(slot, name.clone());
                    }
                    let Some(op) = op.asm_operand() else { continue };
                    (
                        name.map(Name::new_symbol_root),
                        match op {
                            ast::AsmOperand::AsmRegOperand(op) => {
                                let Some(dir_spec) = op.asm_dir_spec() else {
                                    continue;
                                };
                                let Some(reg) = lower_reg(op.asm_reg_spec()) else {
                                    continue;
                                };
                                if dir_spec.in_token().is_some() {
                                    let expr = self.collect_expr_opt(
                                        op.asm_operand_expr().and_then(|it| it.in_expr()),
                                    );
                                    AsmOperand::In { reg, expr }
                                } else if dir_spec.out_token().is_some() {
                                    let expr = op
                                        .asm_operand_expr()
                                        .and_then(|it| it.in_expr())
                                        .filter(|it| !matches!(it, ast::Expr::UnderscoreExpr(_)))
                                        .map(|expr| self.collect_expr(expr));
                                    AsmOperand::Out { reg, expr, late: false }
                                } else if dir_spec.lateout_token().is_some() {
                                    let expr = op
                                        .asm_operand_expr()
                                        .and_then(|it| it.in_expr())
                                        .filter(|it| !matches!(it, ast::Expr::UnderscoreExpr(_)))
                                        .map(|expr| self.collect_expr(expr));

                                    AsmOperand::Out { reg, expr, late: true }
                                } else if dir_spec.inout_token().is_some() {
                                    let Some(op_expr) = op.asm_operand_expr() else { continue };
                                    let in_expr = self.collect_expr_opt(op_expr.in_expr());
                                    match op_expr.fat_arrow_token().is_some() {
                                        true => {
                                            let out_expr = op_expr
                                                .out_expr()
                                                .filter(|it| {
                                                    !matches!(it, ast::Expr::UnderscoreExpr(_))
                                                })
                                                .map(|expr| self.collect_expr(expr));

                                            AsmOperand::SplitInOut {
                                                reg,
                                                in_expr,
                                                out_expr,
                                                late: false,
                                            }
                                        }
                                        false => {
                                            AsmOperand::InOut { reg, expr: in_expr, late: false }
                                        }
                                    }
                                } else if dir_spec.inlateout_token().is_some() {
                                    let Some(op_expr) = op.asm_operand_expr() else { continue };
                                    let in_expr = self.collect_expr_opt(op_expr.in_expr());
                                    match op_expr.fat_arrow_token().is_some() {
                                        true => {
                                            let out_expr = op_expr
                                                .out_expr()
                                                .filter(|it| {
                                                    !matches!(it, ast::Expr::UnderscoreExpr(_))
                                                })
                                                .map(|expr| self.collect_expr(expr));

                                            AsmOperand::SplitInOut {
                                                reg,
                                                in_expr,
                                                out_expr,
                                                late: true,
                                            }
                                        }
                                        false => {
                                            AsmOperand::InOut { reg, expr: in_expr, late: true }
                                        }
                                    }
                                } else {
                                    continue;
                                }
                            }
                            ast::AsmOperand::AsmLabel(l) => {
                                AsmOperand::Label(self.collect_block_opt(l.block_expr()))
                            }
                            ast::AsmOperand::AsmConst(c) => {
                                AsmOperand::Const(self.collect_expr_opt(c.expr()))
                            }
                            ast::AsmOperand::AsmSym(s) => {
                                let Some(path) = s.path().and_then(|p| {
                                    self.lower_path(
                                        p,
                                        &mut ExprCollector::impl_trait_error_allocator,
                                    )
                                }) else {
                                    continue;
                                };
                                AsmOperand::Sym(path)
                            }
                        },
                    )
                }
            };
            operands.push(op);
        }

        let mut mappings = vec![];
        let mut curarg = 0;
        if !options.contains(AsmOptions::RAW) {
            // Don't treat raw asm as a format string.
            asm.template()
                .enumerate()
                .filter_map(|(idx, it)| Some((idx, it.clone(), self.expand_macros_to_string(it)?)))
                .for_each(|(idx, expr, (s, is_direct_literal))| {
                    mappings.resize_with(idx + 1, Vec::default);
                    let Ok(text) = s.value() else {
                        return;
                    };
                    let mappings = &mut mappings[idx];
                    let template_snippet = match expr {
                        ast::Expr::Literal(literal) => match literal.kind() {
                            ast::LiteralKind::String(s) => Some(s.text().to_owned()),
                            _ => None,
                        },
                        _ => None,
                    };
                    let str_style = match s.quote_offsets() {
                        Some(offsets) => {
                            let raw = usize::from(offsets.quotes.0.len()) - 1;
                            // subtract 1 for the `r` prefix
                            (raw != 0).then(|| raw - 1)
                        }
                        None => None,
                    };

                    let mut parser = rustc_parse_format::Parser::new(
                        &text,
                        str_style,
                        template_snippet,
                        false,
                        rustc_parse_format::ParseMode::InlineAsm,
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

                    curarg = parser.curarg;

                    let to_span = |inner_span: std::ops::Range<usize>| {
                        is_direct_literal.then(|| {
                            TextRange::new(
                                inner_span.start.try_into().unwrap(),
                                inner_span.end.try_into().unwrap(),
                            )
                        })
                    };
                    for piece in unverified_pieces {
                        match piece {
                            rustc_parse_format::Piece::Lit(_) => {}
                            rustc_parse_format::Piece::NextArgument(arg) => {
                                // let span = arg_spans.next();

                                let (operand_idx, _name) = match arg.position {
                                    rustc_parse_format::ArgumentIs(idx)
                                    | rustc_parse_format::ArgumentImplicitlyIs(idx) => {
                                        if idx >= operands.len()
                                            || named_pos.contains_key(&idx)
                                            || reg_args.contains(&idx)
                                        {
                                            (None, None)
                                        } else {
                                            (Some(idx), None)
                                        }
                                    }
                                    rustc_parse_format::ArgumentNamed(name) => {
                                        let name = Symbol::intern(name);
                                        (
                                            named_args.get(&name).copied(),
                                            Some(Name::new_symbol_root(name)),
                                        )
                                    }
                                };

                                if let Some(operand_idx) = operand_idx
                                    && let Some(position_span) = to_span(arg.position_span)
                                {
                                    mappings.push((position_span, operand_idx));
                                }
                            }
                        }
                    }
                })
        };

        let kind = if asm.global_asm_token().is_some() {
            InlineAsmKind::GlobalAsm
        } else if asm.naked_asm_token().is_some() {
            InlineAsmKind::NakedAsm
        } else {
            InlineAsmKind::Asm
        };

        let idx = self.alloc_expr(
            Expr::InlineAsm(InlineAsm { operands: operands.into_boxed_slice(), options, kind }),
            syntax_ptr,
        );
        self.store
            .template_map
            .get_or_insert_with(Default::default)
            .asm_to_captures
            .insert(idx, mappings);
        idx
    }
}
