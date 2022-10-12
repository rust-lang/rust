use super::*;
use rustc_ast as ast;
use rustc_ast::visit::{self, Visitor};
use rustc_ast::{BlockCheckMode, UnsafeSource};
use rustc_data_structures::fx::FxIndexSet;
use rustc_span::{sym, symbol::kw};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
enum ArgumentType {
    Format(FormatTrait),
    Usize,
}

fn make_argument(ecx: &ExtCtxt<'_>, sp: Span, arg: P<ast::Expr>, ty: ArgumentType) -> P<ast::Expr> {
    // Generate:
    //     ::core::fmt::ArgumentV1::new_…(arg)
    use ArgumentType::*;
    use FormatTrait::*;
    ecx.expr_call_global(
        sp,
        ecx.std_path(&[
            sym::fmt,
            sym::ArgumentV1,
            match ty {
                Format(Display) => sym::new_display,
                Format(Debug) => sym::new_debug,
                Format(LowerExp) => sym::new_lower_exp,
                Format(UpperExp) => sym::new_upper_exp,
                Format(Octal) => sym::new_octal,
                Format(Pointer) => sym::new_pointer,
                Format(Binary) => sym::new_binary,
                Format(LowerHex) => sym::new_lower_hex,
                Format(UpperHex) => sym::new_upper_hex,
                Usize => sym::from_usize,
            },
        ]),
        vec![arg],
    )
}

fn make_count(
    ecx: &ExtCtxt<'_>,
    sp: Span,
    count: &Option<FormatCount>,
    argmap: &mut FxIndexSet<(usize, ArgumentType)>,
) -> P<ast::Expr> {
    // Generate:
    //     ::core::fmt::rt::v1::Count::…(…)
    match count {
        Some(FormatCount::Literal(n)) => ecx.expr_call_global(
            sp,
            ecx.std_path(&[sym::fmt, sym::rt, sym::v1, sym::Count, sym::Is]),
            vec![ecx.expr_usize(sp, *n)],
        ),
        Some(FormatCount::Argument(arg)) => {
            if let Ok(arg_index) = arg.index {
                let (i, _) = argmap.insert_full((arg_index, ArgumentType::Usize));
                ecx.expr_call_global(
                    sp,
                    ecx.std_path(&[sym::fmt, sym::rt, sym::v1, sym::Count, sym::Param]),
                    vec![ecx.expr_usize(sp, i)],
                )
            } else {
                DummyResult::raw_expr(sp, true)
            }
        }
        None => ecx.expr_path(ecx.path_global(
            sp,
            ecx.std_path(&[sym::fmt, sym::rt, sym::v1, sym::Count, sym::Implied]),
        )),
    }
}

fn make_format_spec(
    ecx: &ExtCtxt<'_>,
    sp: Span,
    placeholder: &FormatPlaceholder,
    argmap: &mut FxIndexSet<(usize, ArgumentType)>,
) -> P<ast::Expr> {
    // Generate:
    //     ::core::fmt::rt::v1::Argument {
    //         position: 0usize,
    //         format: ::core::fmt::rt::v1::FormatSpec {
    //             fill: ' ',
    //             align: ::core::fmt::rt::v1::Alignment::Unknown,
    //             flags: 0u32,
    //             precision: ::core::fmt::rt::v1::Count::Implied,
    //             width: ::core::fmt::rt::v1::Count::Implied,
    //         },
    //     }
    let position = match placeholder.argument.index {
        Ok(arg_index) => {
            let (i, _) =
                argmap.insert_full((arg_index, ArgumentType::Format(placeholder.format_trait)));
            ecx.expr_usize(sp, i)
        }
        Err(_) => DummyResult::raw_expr(sp, true),
    };
    let fill = ecx.expr_char(sp, placeholder.format_options.fill.unwrap_or(' '));
    let align = ecx.expr_path(ecx.path_global(
        sp,
        ecx.std_path(&[
            sym::fmt,
            sym::rt,
            sym::v1,
            sym::Alignment,
            match placeholder.format_options.alignment {
                Some(FormatAlignment::Left) => sym::Left,
                Some(FormatAlignment::Right) => sym::Right,
                Some(FormatAlignment::Center) => sym::Center,
                None => sym::Unknown,
            },
        ]),
    ));
    let flags = ecx.expr_u32(sp, placeholder.format_options.flags);
    let prec = make_count(ecx, sp, &placeholder.format_options.precision, argmap);
    let width = make_count(ecx, sp, &placeholder.format_options.width, argmap);
    ecx.expr_struct(
        sp,
        ecx.path_global(sp, ecx.std_path(&[sym::fmt, sym::rt, sym::v1, sym::Argument])),
        vec![
            ecx.field_imm(sp, Ident::new(sym::position, sp), position),
            ecx.field_imm(
                sp,
                Ident::new(sym::format, sp),
                ecx.expr_struct(
                    sp,
                    ecx.path_global(
                        sp,
                        ecx.std_path(&[sym::fmt, sym::rt, sym::v1, sym::FormatSpec]),
                    ),
                    vec![
                        ecx.field_imm(sp, Ident::new(sym::fill, sp), fill),
                        ecx.field_imm(sp, Ident::new(sym::align, sp), align),
                        ecx.field_imm(sp, Ident::new(sym::flags, sp), flags),
                        ecx.field_imm(sp, Ident::new(sym::precision, sp), prec),
                        ecx.field_imm(sp, Ident::new(sym::width, sp), width),
                    ],
                ),
            ),
        ],
    )
}

pub fn expand_parsed_format_args(ecx: &mut ExtCtxt<'_>, fmt: FormatArgs) -> P<ast::Expr> {
    let macsp = ecx.with_def_site_ctxt(ecx.call_site());

    let lit_pieces = ecx.expr_array_ref(
        fmt.span,
        fmt.template
            .iter()
            .enumerate()
            .filter_map(|(i, piece)| match piece {
                &FormatArgsPiece::Literal(s) => Some(ecx.expr_str(fmt.span, s)),
                &FormatArgsPiece::Placeholder(_) => {
                    // Inject empty string before placeholders when not already preceded by a literal piece.
                    if i == 0 || matches!(fmt.template[i - 1], FormatArgsPiece::Placeholder(_)) {
                        Some(ecx.expr_str(fmt.span, kw::Empty))
                    } else {
                        None
                    }
                }
            })
            .collect(),
    );

    // Whether we'll use the `Arguments::new_v1_formatted` form (true),
    // or the `Arguments::new_v1` form (false).
    let mut use_format_options = false;

    // Create a list of all _unique_ (argument, format trait) combinations.
    // E.g. "{0} {0:x} {0} {1}" -> [(0, Display), (0, LowerHex), (1, Display)]
    let mut argmap = FxIndexSet::default();
    for piece in &fmt.template {
        let FormatArgsPiece::Placeholder(placeholder) = piece else { continue };
        if placeholder.format_options != Default::default() {
            // Can't use basic form if there's any formatting options.
            use_format_options = true;
        }
        if let Ok(index) = placeholder.argument.index {
            if !argmap.insert((index, ArgumentType::Format(placeholder.format_trait))) {
                // Duplicate (argument, format trait) combination,
                // which we'll only put once in the args array.
                use_format_options = true;
            }
        }
    }

    let format_options = use_format_options.then(|| {
        // Generate:
        //     &[format_spec_0, format_spec_1, format_spec_2]
        ecx.expr_array_ref(
            macsp,
            fmt.template
                .iter()
                .filter_map(|piece| {
                    let FormatArgsPiece::Placeholder(placeholder) = piece else { return None };
                    Some(make_format_spec(ecx, macsp, placeholder, &mut argmap))
                })
                .collect(),
        )
    });

    let arguments = fmt.arguments.into_vec();

    // If the args array contains exactly all the original arguments once,
    // in order, we can use a simple array instead of a `match` construction.
    // However, if there's a yield point in any argument except the first one,
    // we don't do this, because an ArgumentV1 cannot be kept across yield points.
    let use_simple_array = argmap.len() == arguments.len()
        && argmap.iter().enumerate().all(|(i, &(j, _))| i == j)
        && arguments.iter().skip(1).all(|arg| !may_contain_yield_point(&arg.expr));

    let args = if use_simple_array {
        // Generate:
        //     &[
        //         ::core::fmt::ArgumentV1::new_display(&arg0),
        //         ::core::fmt::ArgumentV1::new_lower_hex(&arg1),
        //         ::core::fmt::ArgumentV1::new_debug(&arg2),
        //     ]
        ecx.expr_array_ref(
            macsp,
            arguments
                .into_iter()
                .zip(argmap)
                .map(|(arg, (_, ty))| {
                    let sp = arg.expr.span.with_ctxt(macsp.ctxt());
                    make_argument(ecx, sp, ecx.expr_addr_of(sp, arg.expr), ty)
                })
                .collect(),
        )
    } else {
        // Generate:
        //     match (&arg0, &arg1, &arg2) {
        //         args => &[
        //             ::core::fmt::ArgumentV1::new_display(args.0),
        //             ::core::fmt::ArgumentV1::new_lower_hex(args.1),
        //             ::core::fmt::ArgumentV1::new_debug(args.0),
        //         ]
        //     }
        let args_ident = Ident::new(sym::args, macsp);
        let args = argmap
            .iter()
            .map(|&(arg_index, ty)| {
                if let Some(arg) = arguments.get(arg_index) {
                    let sp = arg.expr.span.with_ctxt(macsp.ctxt());
                    make_argument(
                        ecx,
                        sp,
                        ecx.expr_field(
                            sp,
                            ecx.expr_ident(macsp, args_ident),
                            Ident::new(sym::integer(arg_index), macsp),
                        ),
                        ty,
                    )
                } else {
                    DummyResult::raw_expr(macsp, true)
                }
            })
            .collect();
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
                vec![ecx.arm(macsp, ecx.pat_ident(macsp, args_ident), ecx.expr_array(macsp, args))],
            ),
        )
    };

    if let Some(format_options) = format_options {
        // Generate:
        //     ::core::fmt::Arguments::new_v1_formatted(
        //         lit_pieces,
        //         args,
        //         format_options,
        //         unsafe { ::core::fmt::UnsafeArg::new() }
        //     )
        ecx.expr_call_global(
            macsp,
            ecx.std_path(&[sym::fmt, sym::Arguments, sym::new_v1_formatted]),
            vec![
                lit_pieces,
                args,
                format_options,
                ecx.expr_block(P(ast::Block {
                    stmts: vec![ecx.stmt_expr(ecx.expr_call_global(
                        macsp,
                        ecx.std_path(&[sym::fmt, sym::UnsafeArg, sym::new]),
                        Vec::new(),
                    ))],
                    id: ast::DUMMY_NODE_ID,
                    rules: BlockCheckMode::Unsafe(UnsafeSource::CompilerGenerated),
                    span: macsp,
                    tokens: None,
                    could_be_bare_literal: false,
                })),
            ],
        )
    } else {
        // Generate:
        //     ::core::fmt::Arguments::new_v1(
        //         lit_pieces,
        //         args,
        //     )
        ecx.expr_call_global(
            macsp,
            ecx.std_path(&[sym::fmt, sym::Arguments, sym::new_v1]),
            vec![lit_pieces, args],
        )
    }
}

fn may_contain_yield_point(e: &ast::Expr) -> bool {
    struct MayContainYieldPoint(bool);

    impl Visitor<'_> for MayContainYieldPoint {
        fn visit_expr(&mut self, e: &ast::Expr) {
            if let ast::ExprKind::Await(_) | ast::ExprKind::Yield(_) = e.kind {
                self.0 = true;
            } else {
                visit::walk_expr(self, e);
            }
        }

        fn visit_mac_call(&mut self, _: &ast::MacCall) {
            self.0 = true;
        }

        fn visit_attribute(&mut self, _: &ast::Attribute) {
            // Conservatively assume this may be a proc macro attribute in
            // expression position.
            self.0 = true;
        }

        fn visit_item(&mut self, _: &ast::Item) {
            // Do not recurse into nested items.
        }
    }

    let mut visitor = MayContainYieldPoint(false);
    visitor.visit_expr(e);
    visitor.0
}
