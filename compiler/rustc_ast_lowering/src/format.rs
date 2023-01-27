use super::LoweringContext;
use rustc_ast as ast;
use rustc_ast::visit::{self, Visitor};
use rustc_ast::*;
use rustc_data_structures::fx::FxIndexSet;
use rustc_hir as hir;
use rustc_span::{
    sym,
    symbol::{kw, Ident},
    Span,
};

impl<'hir> LoweringContext<'_, 'hir> {
    pub(crate) fn lower_format_args(&mut self, sp: Span, fmt: &FormatArgs) -> hir::ExprKind<'hir> {
        expand_format_args(self, sp, fmt)
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
enum ArgumentType {
    Format(FormatTrait),
    Usize,
}

/// Generate a hir expression representing an argument to a format_args invocation.
///
/// Generates:
///
/// ```text
///     <core::fmt::ArgumentV1>::new_…(arg)
/// ```
fn make_argument<'hir>(
    ctx: &mut LoweringContext<'_, 'hir>,
    sp: Span,
    arg: &'hir hir::Expr<'hir>,
    ty: ArgumentType,
) -> hir::Expr<'hir> {
    use ArgumentType::*;
    use FormatTrait::*;
    let new_fn = ctx.arena.alloc(ctx.expr_lang_item_type_relative(
        sp,
        hir::LangItem::FormatArgument,
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
    ));
    ctx.expr_call_mut(sp, new_fn, std::slice::from_ref(arg))
}

/// Generate a hir expression for a format_args Count.
///
/// Generates:
///
/// ```text
///     <core::fmt::rt::v1::Count>::Is(…)
/// ```
///
/// or
///
/// ```text
///     <core::fmt::rt::v1::Count>::Param(…)
/// ```
///
/// or
///
/// ```text
///     <core::fmt::rt::v1::Count>::Implied
/// ```
fn make_count<'hir>(
    ctx: &mut LoweringContext<'_, 'hir>,
    sp: Span,
    count: &Option<FormatCount>,
    argmap: &mut FxIndexSet<(usize, ArgumentType)>,
) -> hir::Expr<'hir> {
    match count {
        Some(FormatCount::Literal(n)) => {
            let count_is = ctx.arena.alloc(ctx.expr_lang_item_type_relative(
                sp,
                hir::LangItem::FormatCount,
                sym::Is,
            ));
            let value = ctx.arena.alloc_from_iter([ctx.expr_usize(sp, *n)]);
            ctx.expr_call_mut(sp, count_is, value)
        }
        Some(FormatCount::Argument(arg)) => {
            if let Ok(arg_index) = arg.index {
                let (i, _) = argmap.insert_full((arg_index, ArgumentType::Usize));
                let count_param = ctx.arena.alloc(ctx.expr_lang_item_type_relative(
                    sp,
                    hir::LangItem::FormatCount,
                    sym::Param,
                ));
                let value = ctx.arena.alloc_from_iter([ctx.expr_usize(sp, i)]);
                ctx.expr_call_mut(sp, count_param, value)
            } else {
                ctx.expr(sp, hir::ExprKind::Err)
            }
        }
        None => ctx.expr_lang_item_type_relative(sp, hir::LangItem::FormatCount, sym::Implied),
    }
}

/// Generate a hir expression for a format_args placeholder specification.
///
/// Generates
///
/// ```text
///     <core::fmt::rt::v1::Argument::new(
///         …usize, // position
///         '…', // fill
///         <core::fmt::rt::v1::Alignment>::…, // alignment
///         …u32, // flags
///         <core::fmt::rt::v1::Count::…>, // width
///         <core::fmt::rt::v1::Count::…>, // precision
///     )
/// ```
fn make_format_spec<'hir>(
    ctx: &mut LoweringContext<'_, 'hir>,
    sp: Span,
    placeholder: &FormatPlaceholder,
    argmap: &mut FxIndexSet<(usize, ArgumentType)>,
) -> hir::Expr<'hir> {
    let position = match placeholder.argument.index {
        Ok(arg_index) => {
            let (i, _) =
                argmap.insert_full((arg_index, ArgumentType::Format(placeholder.format_trait)));
            ctx.expr_usize(sp, i)
        }
        Err(_) => ctx.expr(sp, hir::ExprKind::Err),
    };
    let &FormatOptions {
        ref width,
        ref precision,
        alignment,
        fill,
        sign,
        alternate,
        zero_pad,
        debug_hex,
    } = &placeholder.format_options;
    let fill = ctx.expr_char(sp, fill.unwrap_or(' '));
    let align = ctx.expr_lang_item_type_relative(
        sp,
        hir::LangItem::FormatAlignment,
        match alignment {
            Some(FormatAlignment::Left) => sym::Left,
            Some(FormatAlignment::Right) => sym::Right,
            Some(FormatAlignment::Center) => sym::Center,
            None => sym::Unknown,
        },
    );
    // This needs to match `FlagV1` in library/core/src/fmt/mod.rs.
    let flags: u32 = ((sign == Some(FormatSign::Plus)) as u32)
        | ((sign == Some(FormatSign::Minus)) as u32) << 1
        | (alternate as u32) << 2
        | (zero_pad as u32) << 3
        | ((debug_hex == Some(FormatDebugHex::Lower)) as u32) << 4
        | ((debug_hex == Some(FormatDebugHex::Upper)) as u32) << 5;
    let flags = ctx.expr_u32(sp, flags);
    let precision = make_count(ctx, sp, &precision, argmap);
    let width = make_count(ctx, sp, &width, argmap);
    let format_placeholder_new = ctx.arena.alloc(ctx.expr_lang_item_type_relative(
        sp,
        hir::LangItem::FormatPlaceholder,
        sym::new,
    ));
    let args = ctx.arena.alloc_from_iter([position, fill, align, flags, precision, width]);
    ctx.expr_call_mut(sp, format_placeholder_new, args)
}

fn expand_format_args<'hir>(
    ctx: &mut LoweringContext<'_, 'hir>,
    macsp: Span,
    fmt: &FormatArgs,
) -> hir::ExprKind<'hir> {
    let lit_pieces =
        ctx.arena.alloc_from_iter(fmt.template.iter().enumerate().filter_map(|(i, piece)| {
            match piece {
                &FormatArgsPiece::Literal(s) => Some(ctx.expr_str(fmt.span, s)),
                &FormatArgsPiece::Placeholder(_) => {
                    // Inject empty string before placeholders when not already preceded by a literal piece.
                    if i == 0 || matches!(fmt.template[i - 1], FormatArgsPiece::Placeholder(_)) {
                        Some(ctx.expr_str(fmt.span, kw::Empty))
                    } else {
                        None
                    }
                }
            }
        }));
    let lit_pieces = ctx.expr_array_ref(fmt.span, lit_pieces);

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
        let elements: Vec<_> = fmt
            .template
            .iter()
            .filter_map(|piece| {
                let FormatArgsPiece::Placeholder(placeholder) = piece else { return None };
                Some(make_format_spec(ctx, macsp, placeholder, &mut argmap))
            })
            .collect();
        ctx.expr_array_ref(macsp, ctx.arena.alloc_from_iter(elements))
    });

    let arguments = fmt.arguments.all_args();

    // If the args array contains exactly all the original arguments once,
    // in order, we can use a simple array instead of a `match` construction.
    // However, if there's a yield point in any argument except the first one,
    // we don't do this, because an ArgumentV1 cannot be kept across yield points.
    //
    // This is an optimization, speeding up compilation about 1-2% in some cases.
    // See https://github.com/rust-lang/rust/pull/106770#issuecomment-1380790609
    let use_simple_array = argmap.len() == arguments.len()
        && argmap.iter().enumerate().all(|(i, &(j, _))| i == j)
        && arguments.iter().skip(1).all(|arg| !may_contain_yield_point(&arg.expr));

    let args = if use_simple_array {
        // Generate:
        //     &[
        //         <core::fmt::ArgumentV1>::new_display(&arg0),
        //         <core::fmt::ArgumentV1>::new_lower_hex(&arg1),
        //         <core::fmt::ArgumentV1>::new_debug(&arg2),
        //         …
        //     ]
        let elements: Vec<_> = arguments
            .iter()
            .zip(argmap)
            .map(|(arg, (_, ty))| {
                let sp = arg.expr.span.with_ctxt(macsp.ctxt());
                let arg = ctx.lower_expr(&arg.expr);
                let ref_arg = ctx.arena.alloc(ctx.expr(
                    sp,
                    hir::ExprKind::AddrOf(hir::BorrowKind::Ref, hir::Mutability::Not, arg),
                ));
                make_argument(ctx, sp, ref_arg, ty)
            })
            .collect();
        ctx.expr_array_ref(macsp, ctx.arena.alloc_from_iter(elements))
    } else {
        // Generate:
        //     &match (&arg0, &arg1, &…) {
        //         args => [
        //             <core::fmt::ArgumentV1>::new_display(args.0),
        //             <core::fmt::ArgumentV1>::new_lower_hex(args.1),
        //             <core::fmt::ArgumentV1>::new_debug(args.0),
        //             …
        //         ]
        //     }
        let args_ident = Ident::new(sym::args, macsp);
        let (args_pat, args_hir_id) = ctx.pat_ident(macsp, args_ident);
        let args = ctx.arena.alloc_from_iter(argmap.iter().map(|&(arg_index, ty)| {
            if let Some(arg) = arguments.get(arg_index) {
                let sp = arg.expr.span.with_ctxt(macsp.ctxt());
                let args_ident_expr = ctx.expr_ident(macsp, args_ident, args_hir_id);
                let arg = ctx.arena.alloc(ctx.expr(
                    sp,
                    hir::ExprKind::Field(
                        args_ident_expr,
                        Ident::new(sym::integer(arg_index), macsp),
                    ),
                ));
                make_argument(ctx, sp, arg, ty)
            } else {
                ctx.expr(macsp, hir::ExprKind::Err)
            }
        }));
        let elements: Vec<_> = arguments
            .iter()
            .map(|arg| {
                let arg_expr = ctx.lower_expr(&arg.expr);
                ctx.expr(
                    arg.expr.span.with_ctxt(macsp.ctxt()),
                    hir::ExprKind::AddrOf(hir::BorrowKind::Ref, hir::Mutability::Not, arg_expr),
                )
            })
            .collect();
        let args_tuple = ctx
            .arena
            .alloc(ctx.expr(macsp, hir::ExprKind::Tup(ctx.arena.alloc_from_iter(elements))));
        let array = ctx.arena.alloc(ctx.expr(macsp, hir::ExprKind::Array(args)));
        let match_arms = ctx.arena.alloc_from_iter([ctx.arm(args_pat, array)]);
        let match_expr = ctx.arena.alloc(ctx.expr_match(
            macsp,
            args_tuple,
            match_arms,
            hir::MatchSource::FormatArgs,
        ));
        ctx.expr(
            macsp,
            hir::ExprKind::AddrOf(hir::BorrowKind::Ref, hir::Mutability::Not, match_expr),
        )
    };

    if let Some(format_options) = format_options {
        // Generate:
        //     <core::fmt::Arguments>::new_v1_formatted(
        //         lit_pieces,
        //         args,
        //         format_options,
        //         unsafe { ::core::fmt::UnsafeArg::new() }
        //     )
        let new_v1_formatted = ctx.arena.alloc(ctx.expr_lang_item_type_relative(
            macsp,
            hir::LangItem::FormatArguments,
            sym::new_v1_formatted,
        ));
        let unsafe_arg_new = ctx.arena.alloc(ctx.expr_lang_item_type_relative(
            macsp,
            hir::LangItem::FormatUnsafeArg,
            sym::new,
        ));
        let unsafe_arg_new_call = ctx.expr_call(macsp, unsafe_arg_new, &[]);
        let hir_id = ctx.next_id();
        let unsafe_arg = ctx.expr_block(ctx.arena.alloc(hir::Block {
            stmts: &[],
            expr: Some(unsafe_arg_new_call),
            hir_id,
            rules: hir::BlockCheckMode::UnsafeBlock(hir::UnsafeSource::CompilerGenerated),
            span: macsp,
            targeted_by_break: false,
        }));
        let args = ctx.arena.alloc_from_iter([lit_pieces, args, format_options, unsafe_arg]);
        hir::ExprKind::Call(new_v1_formatted, args)
    } else {
        // Generate:
        //     <core::fmt::Arguments>::new_v1(
        //         lit_pieces,
        //         args,
        //     )
        let new_v1 = ctx.arena.alloc(ctx.expr_lang_item_type_relative(
            macsp,
            hir::LangItem::FormatArguments,
            sym::new_v1,
        ));
        let new_args = ctx.arena.alloc_from_iter([lit_pieces, args]);
        hir::ExprKind::Call(new_v1, new_args)
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
            // Macros should be expanded at this point.
            unreachable!("unexpanded macro in ast lowering");
        }

        fn visit_item(&mut self, _: &ast::Item) {
            // Do not recurse into nested items.
        }
    }

    let mut visitor = MayContainYieldPoint(false);
    visitor.visit_expr(e);
    visitor.0
}
