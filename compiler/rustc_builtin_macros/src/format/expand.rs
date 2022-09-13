use super::*;
use rustc_ast as ast;
use rustc_span::symbol::{kw, sym, Ident};

pub fn expand_parsed_format_args(ecx: &mut ExtCtxt<'_>, fmt: FormatArgs) -> P<ast::Expr> {
    let macsp = ecx.with_def_site_ctxt(ecx.call_site());

    if fmt.template.is_empty() {
        return ecx.expr_call_global(
            macsp,
            ecx.std_path(&[sym::fmt, sym::Arguments, sym::from_static_str]),
            vec![ecx.expr_str(macsp, kw::Empty)],
        );
    }

    if let &[FormatArgsPiece::Literal(s)] = &fmt.template[..] {
        return ecx.expr_call_global(
            macsp,
            ecx.std_path(&[sym::fmt, sym::Arguments, sym::from_static_str]),
            vec![ecx.expr_str(macsp, s)],
        );
    }

    let args = Ident::new(sym::_args, macsp);
    let w = Ident::new(sym::w, macsp);

    let arguments = fmt.arguments.into_vec();

    let mut statements = Vec::new();

    for piece in fmt.template {
        match piece {
            FormatArgsPiece::Literal(s) => {
                // Generate:
                //     w.write_str("…")?;
                statements.push(ecx.stmt_expr(ecx.expr(
                    macsp,
                    ast::ExprKind::Try(ecx.expr(
                        macsp,
                        ast::ExprKind::MethodCall(
                            ast::PathSegment::from_ident(Ident::new(sym::write_str, macsp)),
                            ecx.expr_ident(macsp, w),
                            vec![ecx.expr_str(macsp, s)],
                            macsp,
                        ),
                    )),
                )));
            }
            FormatArgsPiece::Placeholder(p) => {
                // Don't set options if they're still set to defaults
                // and this placeholder also uses default options.

                // Generate:
                //     ::core::fmt::Formatter::new(w)
                let mut formatter = ecx.expr_call_global(
                    macsp,
                    ecx.std_path(&[sym::fmt, sym::Formatter, sym::new]),
                    vec![ecx.expr_ident(macsp, w)],
                );

                if p.format_options != FormatOptions::default() {
                    // Add:
                    //     .with_options(…)
                    formatter = ecx.expr(
                        macsp,
                        ast::ExprKind::MethodCall(
                            ast::PathSegment::from_ident(Ident::new(sym::with_options, macsp)),
                            formatter,
                            vec![
                                ecx.expr_u32(macsp, p.format_options.flags),
                                ecx.expr_char(macsp, p.format_options.fill.unwrap_or(' ')),
                                ecx.expr_path(ecx.path_global(
                                    macsp,
                                    ecx.std_path(&[
                                        sym::fmt,
                                        sym::rt,
                                        sym::v1,
                                        sym::Alignment,
                                        match p.format_options.alignment {
                                            Some(FormatAlignment::Left) => sym::Left,
                                            Some(FormatAlignment::Right) => sym::Right,
                                            Some(FormatAlignment::Center) => sym::Center,
                                            None => sym::Unknown,
                                        },
                                    ]),
                                )),
                                make_count(ecx, macsp, &arguments, args, p.format_options.width),
                                make_count(
                                    ecx,
                                    macsp,
                                    &arguments,
                                    args,
                                    p.format_options.precision,
                                ),
                            ],
                            macsp,
                        ),
                    );
                }

                // Generate:
                //     ::core::fmt::Display::fmt(arg.0, &mut formatter)?;

                let arg = if let Ok(i) = p.argument.index {
                    ecx.expr_field(
                        arguments[i].expr.span.with_ctxt(macsp.ctxt()),
                        ecx.expr_ident(macsp, args),
                        Ident::new(sym::integer(i), macsp),
                    )
                } else {
                    DummyResult::raw_expr(macsp, true)
                };
                let fmt_trait = match p.format_trait {
                    FormatTrait::Display => sym::Display,
                    FormatTrait::Debug => sym::Debug,
                    FormatTrait::LowerExp => sym::LowerExp,
                    FormatTrait::UpperExp => sym::UpperExp,
                    FormatTrait::Octal => sym::Octal,
                    FormatTrait::Pointer => sym::Pointer,
                    FormatTrait::Binary => sym::Binary,
                    FormatTrait::LowerHex => sym::LowerHex,
                    FormatTrait::UpperHex => sym::UpperHex,
                };
                statements.push(ecx.stmt_expr(ecx.expr(
                    macsp,
                    ast::ExprKind::Try(ecx.expr_call_global(
                        arg.span,
                        ecx.std_path(&[sym::fmt, fmt_trait, sym::fmt]),
                        vec![
                            arg,
                            ecx.expr(
                                macsp,
                                ast::ExprKind::AddrOf(
                                    ast::BorrowKind::Ref,
                                    ast::Mutability::Mut,
                                    formatter,
                                ),
                            ),
                        ],
                    )),
                )));
            }
        }
    }

    // Generate:
    //     Ok(())
    statements.push(ecx.stmt_expr(ecx.expr_ok(macsp, ecx.expr_tuple(macsp, Vec::new()))));

    // Generate:
    //     |w: &mut dyn ::core::fmt::Write| -> ::core::fmt::Result {
    //         … // statements
    //     }
    let closure = ecx.expr(
        macsp,
        ast::ExprKind::Closure(
            ast::ClosureBinder::NotPresent,
            ast::CaptureBy::Ref,
            ast::Async::No,
            ast::Movability::Movable,
            ecx.fn_decl(
                vec![ecx.param(
                    macsp,
                    w,
                    ecx.ty_rptr(
                        macsp,
                        ecx.ty(
                            macsp,
                            ast::TyKind::TraitObject(
                                vec![ast::GenericBound::Trait(
                                    ast::PolyTraitRef::new(
                                        vec![],
                                        ecx.path_global(
                                            macsp,
                                            ecx.std_path(&[sym::fmt, sym::Write]),
                                        ),
                                        macsp,
                                    ),
                                    ast::TraitBoundModifier::None,
                                )],
                                ast::TraitObjectSyntax::Dyn,
                            ),
                        ),
                        None,
                        ast::Mutability::Mut,
                    ),
                )],
                ast::FnRetTy::Ty(
                    ecx.ty_path(ecx.path_global(macsp, ecx.std_path(&[sym::fmt, sym::Result]))),
                ),
            ),
            ecx.expr_block(ecx.block(macsp, statements)),
            macsp,
        ),
    );

    // Generate:
    //     ::core::fmt::Arguments::new(
    //         &match (&arg0, &arg1, …) {
    //             args => closure,
    //         }
    //     )
    ecx.expr_call_global(
        macsp,
        ecx.std_path(&[sym::fmt, sym::Arguments, sym::new]),
        vec![
            ecx.expr_addr_of(
                macsp,
                ecx.expr_match(
                    macsp,
                    ecx.expr_tuple(
                        macsp,
                        arguments
                            .into_iter()
                            .map(|arg| {
                                ecx.expr_addr_of(arg.expr.span.with_ctxt(macsp.ctxt()), arg.expr)
                            })
                            .collect(),
                    ),
                    vec![ecx.arm(macsp, ecx.pat_ident(macsp, args), closure)],
                ),
            ),
        ],
    )
}

pub fn make_count(
    ecx: &ExtCtxt<'_>,
    macsp: Span,
    arguments: &[FormatArgument],
    args: Ident,
    count: Option<FormatCount>,
) -> P<ast::Expr> {
    match count {
        Some(FormatCount::Literal(n)) => ecx.expr_some(macsp, ecx.expr_usize(macsp, n)),
        Some(FormatCount::Argument(arg)) => {
            if let Ok(i) = arg.index {
                let sp = arguments[i].expr.span.with_ctxt(macsp.ctxt());
                ecx.expr_some(
                    sp,
                    ecx.expr_deref(
                        sp,
                        ecx.expr_field(
                            sp,
                            ecx.expr_ident(macsp, args),
                            Ident::new(sym::integer(i), macsp),
                        ),
                    ),
                )
            } else {
                DummyResult::raw_expr(macsp, true)
            }
        }
        None => ecx.expr_none(macsp),
    }
}
