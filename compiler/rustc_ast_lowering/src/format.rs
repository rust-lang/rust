use core::ops::ControlFlow;
use std::borrow::Cow;

use rustc_ast::visit::Visitor;
use rustc_ast::*;
use rustc_data_structures::fx::FxIndexMap;
use rustc_hir as hir;
use rustc_session::config::FmtDebug;
use rustc_span::{Ident, Span, Symbol, kw, sym};

use super::LoweringContext;

impl<'hir> LoweringContext<'_, 'hir> {
    pub(crate) fn lower_format_args(&mut self, sp: Span, fmt: &FormatArgs) -> hir::ExprKind<'hir> {
        // Never call the const constructor of `fmt::Arguments` if the
        // format_args!() had any arguments _before_ flattening/inlining.
        let allow_const = fmt.arguments.all_args().is_empty();
        let mut fmt = Cow::Borrowed(fmt);
        if self.tcx.sess.opts.unstable_opts.flatten_format_args {
            fmt = flatten_format_args(fmt);
            fmt = self.inline_literals(fmt);
        }
        expand_format_args(self, sp, &fmt, allow_const)
    }

    /// Try to convert a literal into an interned string
    fn try_inline_lit(&self, lit: token::Lit) -> Option<Symbol> {
        match LitKind::from_token_lit(lit) {
            Ok(LitKind::Str(s, _)) => Some(s),
            Ok(LitKind::Int(n, ty)) => {
                match ty {
                    // unsuffixed integer literals are assumed to be i32's
                    LitIntType::Unsuffixed => {
                        (n <= i32::MAX as u128).then_some(Symbol::intern(&n.to_string()))
                    }
                    LitIntType::Signed(int_ty) => {
                        let max_literal = self.int_ty_max(int_ty);
                        (n <= max_literal).then_some(Symbol::intern(&n.to_string()))
                    }
                    LitIntType::Unsigned(uint_ty) => {
                        let max_literal = self.uint_ty_max(uint_ty);
                        (n <= max_literal).then_some(Symbol::intern(&n.to_string()))
                    }
                }
            }
            _ => None,
        }
    }

    /// Get the maximum value of int_ty. It is platform-dependent due to the byte size of isize
    fn int_ty_max(&self, int_ty: IntTy) -> u128 {
        match int_ty {
            IntTy::Isize => self.tcx.data_layout.pointer_size.signed_int_max() as u128,
            IntTy::I8 => i8::MAX as u128,
            IntTy::I16 => i16::MAX as u128,
            IntTy::I32 => i32::MAX as u128,
            IntTy::I64 => i64::MAX as u128,
            IntTy::I128 => i128::MAX as u128,
        }
    }

    /// Get the maximum value of uint_ty. It is platform-dependent due to the byte size of usize
    fn uint_ty_max(&self, uint_ty: UintTy) -> u128 {
        match uint_ty {
            UintTy::Usize => self.tcx.data_layout.pointer_size.unsigned_int_max(),
            UintTy::U8 => u8::MAX as u128,
            UintTy::U16 => u16::MAX as u128,
            UintTy::U32 => u32::MAX as u128,
            UintTy::U64 => u64::MAX as u128,
            UintTy::U128 => u128::MAX as u128,
        }
    }

    /// Inline literals into the format string.
    ///
    /// Turns
    ///
    /// `format_args!("Hello, {}! {} {}", "World", 123, x)`
    ///
    /// into
    ///
    /// `format_args!("Hello, World! 123 {}", x)`.
    fn inline_literals<'fmt>(&self, mut fmt: Cow<'fmt, FormatArgs>) -> Cow<'fmt, FormatArgs> {
        let mut was_inlined = vec![false; fmt.arguments.all_args().len()];
        let mut inlined_anything = false;

        for i in 0..fmt.template.len() {
            let FormatArgsPiece::Placeholder(placeholder) = &fmt.template[i] else { continue };
            let Ok(arg_index) = placeholder.argument.index else { continue };

            let mut literal = None;

            if let FormatTrait::Display = placeholder.format_trait
                && placeholder.format_options == Default::default()
                && let arg = fmt.arguments.all_args()[arg_index].expr.peel_parens_and_refs()
                && let ExprKind::Lit(lit) = arg.kind
            {
                literal = self.try_inline_lit(lit);
            }

            if let Some(literal) = literal {
                // Now we need to mutate the outer FormatArgs.
                // If this is the first time, this clones the outer FormatArgs.
                let fmt = fmt.to_mut();
                // Replace the placeholder with the literal.
                fmt.template[i] = FormatArgsPiece::Literal(literal);
                was_inlined[arg_index] = true;
                inlined_anything = true;
            }
        }

        // Remove the arguments that were inlined.
        if inlined_anything {
            let fmt = fmt.to_mut();

            let mut remove = was_inlined;

            // Don't remove anything that's still used.
            for_all_argument_indexes(&mut fmt.template, |index| remove[*index] = false);

            // Drop all the arguments that are marked for removal.
            let mut remove_it = remove.iter();
            fmt.arguments.all_args_mut().retain(|_| remove_it.next() != Some(&true));

            // Calculate the mapping of old to new indexes for the remaining arguments.
            let index_map: Vec<usize> = remove
                .into_iter()
                .scan(0, |i, remove| {
                    let mapped = *i;
                    *i += !remove as usize;
                    Some(mapped)
                })
                .collect();

            // Correct the indexes that refer to arguments that have shifted position.
            for_all_argument_indexes(&mut fmt.template, |index| *index = index_map[*index]);
        }

        fmt
    }
}

/// Flattens nested `format_args!()` into one.
///
/// Turns
///
/// `format_args!("a {} {} {}.", 1, format_args!("b{}!", 2), 3)`
///
/// into
///
/// `format_args!("a {} b{}! {}.", 1, 2, 3)`.
fn flatten_format_args(mut fmt: Cow<'_, FormatArgs>) -> Cow<'_, FormatArgs> {
    let mut i = 0;
    while i < fmt.template.len() {
        if let FormatArgsPiece::Placeholder(placeholder) = &fmt.template[i]
            && let FormatTrait::Display | FormatTrait::Debug = &placeholder.format_trait
            && let Ok(arg_index) = placeholder.argument.index
            && let arg = fmt.arguments.all_args()[arg_index].expr.peel_parens_and_refs()
            && let ExprKind::FormatArgs(_) = &arg.kind
            // Check that this argument is not used by any other placeholders.
            && fmt.template.iter().enumerate().all(|(j, p)|
                i == j ||
                !matches!(p, FormatArgsPiece::Placeholder(placeholder)
                    if placeholder.argument.index == Ok(arg_index))
            )
        {
            // Now we need to mutate the outer FormatArgs.
            // If this is the first time, this clones the outer FormatArgs.
            let fmt = fmt.to_mut();

            // Take the inner FormatArgs out of the outer arguments, and
            // replace it by the inner arguments. (We can't just put those at
            // the end, because we need to preserve the order of evaluation.)

            let args = fmt.arguments.all_args_mut();
            let remaining_args = args.split_off(arg_index + 1);
            let old_arg_offset = args.len();
            let mut fmt2 = &mut args.pop().unwrap().expr; // The inner FormatArgs.
            let fmt2 = loop {
                // Unwrap the Expr to get to the FormatArgs.
                match &mut fmt2.kind {
                    ExprKind::Paren(inner) | ExprKind::AddrOf(BorrowKind::Ref, _, inner) => {
                        fmt2 = inner
                    }
                    ExprKind::FormatArgs(fmt2) => break fmt2,
                    _ => unreachable!(),
                }
            };

            args.append(fmt2.arguments.all_args_mut());
            let new_arg_offset = args.len();
            args.extend(remaining_args);

            // Correct the indexes that refer to the arguments after the newly inserted arguments.
            for_all_argument_indexes(&mut fmt.template, |index| {
                if *index >= old_arg_offset {
                    *index -= old_arg_offset;
                    *index += new_arg_offset;
                }
            });

            // Now merge the placeholders:

            let rest = fmt.template.split_off(i + 1);
            fmt.template.pop(); // remove the placeholder for the nested fmt args.
            // Insert the pieces from the nested format args, but correct any
            // placeholders to point to the correct argument index.
            for_all_argument_indexes(&mut fmt2.template, |index| *index += arg_index);
            fmt.template.append(&mut fmt2.template);
            fmt.template.extend(rest);

            // Don't increment `i` here, so we recurse into the newly added pieces.
        } else {
            i += 1;
        }
    }
    fmt
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
///     <core::fmt::Argument>::new_…(arg)
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
            Format(Debug) => match ctx.tcx.sess.opts.unstable_opts.fmt_debug {
                FmtDebug::Full | FmtDebug::Shallow => sym::new_debug,
                FmtDebug::None => sym::new_debug_noop,
            },
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
///     <core::fmt::rt::Count>::Is(…)
/// ```
///
/// or
///
/// ```text
///     <core::fmt::rt::Count>::Param(…)
/// ```
///
/// or
///
/// ```text
///     <core::fmt::rt::Count>::Implied
/// ```
fn make_count<'hir>(
    ctx: &mut LoweringContext<'_, 'hir>,
    sp: Span,
    count: &Option<FormatCount>,
    argmap: &mut FxIndexMap<(usize, ArgumentType), Option<Span>>,
) -> hir::Expr<'hir> {
    match count {
        Some(FormatCount::Literal(n)) => {
            let count_is = ctx.arena.alloc(ctx.expr_lang_item_type_relative(
                sp,
                hir::LangItem::FormatCount,
                sym::Is,
            ));
            let value = ctx.arena.alloc_from_iter([ctx.expr_u16(sp, *n)]);
            ctx.expr_call_mut(sp, count_is, value)
        }
        Some(FormatCount::Argument(arg)) => {
            if let Ok(arg_index) = arg.index {
                let (i, _) = argmap.insert_full((arg_index, ArgumentType::Usize), arg.span);
                let count_param = ctx.arena.alloc(ctx.expr_lang_item_type_relative(
                    sp,
                    hir::LangItem::FormatCount,
                    sym::Param,
                ));
                let value = ctx.arena.alloc_from_iter([ctx.expr_usize(sp, i)]);
                ctx.expr_call_mut(sp, count_param, value)
            } else {
                ctx.expr(
                    sp,
                    hir::ExprKind::Err(
                        ctx.dcx().span_delayed_bug(sp, "lowered bad format_args count"),
                    ),
                )
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
///     <core::fmt::rt::Placeholder {
///         position: …usize,
///         flags: …u32,
///         precision: <core::fmt::rt::Count::…>,
///         width: <core::fmt::rt::Count::…>,
///     }
/// ```
fn make_format_spec<'hir>(
    ctx: &mut LoweringContext<'_, 'hir>,
    sp: Span,
    placeholder: &FormatPlaceholder,
    argmap: &mut FxIndexMap<(usize, ArgumentType), Option<Span>>,
) -> hir::Expr<'hir> {
    let position = match placeholder.argument.index {
        Ok(arg_index) => {
            let (i, _) = argmap.insert_full(
                (arg_index, ArgumentType::Format(placeholder.format_trait)),
                placeholder.span,
            );
            ctx.expr_usize(sp, i)
        }
        Err(_) => ctx.expr(
            sp,
            hir::ExprKind::Err(ctx.dcx().span_delayed_bug(sp, "lowered bad format_args count")),
        ),
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
    let fill = fill.unwrap_or(' ');
    // These need to match the constants in library/core/src/fmt/rt.rs.
    let align = match alignment {
        Some(FormatAlignment::Left) => 0,
        Some(FormatAlignment::Right) => 1,
        Some(FormatAlignment::Center) => 2,
        None => 3,
    };
    // This needs to match the constants in library/core/src/fmt/rt.rs.
    let flags: u32 = fill as u32
        | ((sign == Some(FormatSign::Plus)) as u32) << 21
        | ((sign == Some(FormatSign::Minus)) as u32) << 22
        | (alternate as u32) << 23
        | (zero_pad as u32) << 24
        | ((debug_hex == Some(FormatDebugHex::Lower)) as u32) << 25
        | ((debug_hex == Some(FormatDebugHex::Upper)) as u32) << 26
        | (width.is_some() as u32) << 27
        | (precision.is_some() as u32) << 28
        | align << 29
        | 1 << 31; // Highest bit always set.
    let flags = ctx.expr_u32(sp, flags);
    let precision = make_count(ctx, sp, precision, argmap);
    let width = make_count(ctx, sp, width, argmap);
    let position = ctx.expr_field(Ident::new(sym::position, sp), ctx.arena.alloc(position), sp);
    let flags = ctx.expr_field(Ident::new(sym::flags, sp), ctx.arena.alloc(flags), sp);
    let precision = ctx.expr_field(Ident::new(sym::precision, sp), ctx.arena.alloc(precision), sp);
    let width = ctx.expr_field(Ident::new(sym::width, sp), ctx.arena.alloc(width), sp);
    let placeholder = ctx.arena.alloc(hir::QPath::LangItem(hir::LangItem::FormatPlaceholder, sp));
    let fields = ctx.arena.alloc_from_iter([position, flags, precision, width]);
    ctx.expr(sp, hir::ExprKind::Struct(placeholder, fields, hir::StructTailExpr::None))
}

fn expand_format_args<'hir>(
    ctx: &mut LoweringContext<'_, 'hir>,
    macsp: Span,
    fmt: &FormatArgs,
    allow_const: bool,
) -> hir::ExprKind<'hir> {
    let mut incomplete_lit = String::new();
    let lit_pieces =
        ctx.arena.alloc_from_iter(fmt.template.iter().enumerate().filter_map(|(i, piece)| {
            match piece {
                &FormatArgsPiece::Literal(s) => {
                    // Coalesce adjacent literal pieces.
                    if let Some(FormatArgsPiece::Literal(_)) = fmt.template.get(i + 1) {
                        incomplete_lit.push_str(s.as_str());
                        None
                    } else if !incomplete_lit.is_empty() {
                        incomplete_lit.push_str(s.as_str());
                        let s = Symbol::intern(&incomplete_lit);
                        incomplete_lit.clear();
                        Some(ctx.expr_str(fmt.span, s))
                    } else {
                        Some(ctx.expr_str(fmt.span, s))
                    }
                }
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
    let mut argmap = FxIndexMap::default();
    for piece in &fmt.template {
        let FormatArgsPiece::Placeholder(placeholder) = piece else { continue };
        if placeholder.format_options != Default::default() {
            // Can't use basic form if there's any formatting options.
            use_format_options = true;
        }
        if let Ok(index) = placeholder.argument.index {
            if argmap
                .insert((index, ArgumentType::Format(placeholder.format_trait)), placeholder.span)
                .is_some()
            {
                // Duplicate (argument, format trait) combination,
                // which we'll only put once in the args array.
                use_format_options = true;
            }
        }
    }

    let format_options = use_format_options.then(|| {
        // Generate:
        //     &[format_spec_0, format_spec_1, format_spec_2]
        let elements = ctx.arena.alloc_from_iter(fmt.template.iter().filter_map(|piece| {
            let FormatArgsPiece::Placeholder(placeholder) = piece else { return None };
            Some(make_format_spec(ctx, macsp, placeholder, &mut argmap))
        }));
        ctx.expr_array_ref(macsp, elements)
    });

    let arguments = fmt.arguments.all_args();

    if allow_const && arguments.is_empty() && argmap.is_empty() {
        // Generate:
        //     <core::fmt::Arguments>::new_const(lit_pieces)
        let new = ctx.arena.alloc(ctx.expr_lang_item_type_relative(
            macsp,
            hir::LangItem::FormatArguments,
            sym::new_const,
        ));
        let new_args = ctx.arena.alloc_from_iter([lit_pieces]);
        return hir::ExprKind::Call(new, new_args);
    }

    // If the args array contains exactly all the original arguments once,
    // in order, we can use a simple array instead of a `match` construction.
    // However, if there's a yield point in any argument except the first one,
    // we don't do this, because an Argument cannot be kept across yield points.
    //
    // This is an optimization, speeding up compilation about 1-2% in some cases.
    // See https://github.com/rust-lang/rust/pull/106770#issuecomment-1380790609
    let use_simple_array = argmap.len() == arguments.len()
        && argmap.iter().enumerate().all(|(i, (&(j, _), _))| i == j)
        && arguments.iter().skip(1).all(|arg| !may_contain_yield_point(&arg.expr));

    let args = if arguments.is_empty() {
        // Generate:
        //    &<core::fmt::Argument>::none()
        //
        // Note:
        //     `none()` just returns `[]`. We use `none()` rather than `[]` to limit the lifetime.
        //
        //     This makes sure that this still fails to compile, even when the argument is inlined:
        //
        //     ```
        //     let f = format_args!("{}", "a");
        //     println!("{f}"); // error E0716
        //     ```
        //
        //     Cases where keeping the object around is allowed, such as `format_args!("a")`,
        //     are handled above by the `allow_const` case.
        let none_fn = ctx.arena.alloc(ctx.expr_lang_item_type_relative(
            macsp,
            hir::LangItem::FormatArgument,
            sym::none,
        ));
        let none = ctx.expr_call(macsp, none_fn, &[]);
        ctx.expr(macsp, hir::ExprKind::AddrOf(hir::BorrowKind::Ref, hir::Mutability::Not, none))
    } else if use_simple_array {
        // Generate:
        //     &[
        //         <core::fmt::Argument>::new_display(&arg0),
        //         <core::fmt::Argument>::new_lower_hex(&arg1),
        //         <core::fmt::Argument>::new_debug(&arg2),
        //         …
        //     ]
        let elements = ctx.arena.alloc_from_iter(arguments.iter().zip(argmap).map(
            |(arg, ((_, ty), placeholder_span))| {
                let placeholder_span =
                    placeholder_span.unwrap_or(arg.expr.span).with_ctxt(macsp.ctxt());
                let arg_span = match arg.kind {
                    FormatArgumentKind::Captured(_) => placeholder_span,
                    _ => arg.expr.span.with_ctxt(macsp.ctxt()),
                };
                let arg = ctx.lower_expr(&arg.expr);
                let ref_arg = ctx.arena.alloc(ctx.expr(
                    arg_span,
                    hir::ExprKind::AddrOf(hir::BorrowKind::Ref, hir::Mutability::Not, arg),
                ));
                make_argument(ctx, placeholder_span, ref_arg, ty)
            },
        ));
        ctx.expr_array_ref(macsp, elements)
    } else {
        // Generate:
        //     &match (&arg0, &arg1, &…) {
        //         args => [
        //             <core::fmt::Argument>::new_display(args.0),
        //             <core::fmt::Argument>::new_lower_hex(args.1),
        //             <core::fmt::Argument>::new_debug(args.0),
        //             …
        //         ]
        //     }
        let args_ident = Ident::new(sym::args, macsp);
        let (args_pat, args_hir_id) = ctx.pat_ident(macsp, args_ident);
        let args = ctx.arena.alloc_from_iter(argmap.iter().map(
            |(&(arg_index, ty), &placeholder_span)| {
                let arg = &arguments[arg_index];
                let placeholder_span =
                    placeholder_span.unwrap_or(arg.expr.span).with_ctxt(macsp.ctxt());
                let arg_span = match arg.kind {
                    FormatArgumentKind::Captured(_) => placeholder_span,
                    _ => arg.expr.span.with_ctxt(macsp.ctxt()),
                };
                let args_ident_expr = ctx.expr_ident(macsp, args_ident, args_hir_id);
                let arg = ctx.arena.alloc(ctx.expr(
                    arg_span,
                    hir::ExprKind::Field(
                        args_ident_expr,
                        Ident::new(sym::integer(arg_index), macsp),
                    ),
                ));
                make_argument(ctx, placeholder_span, arg, ty)
            },
        ));
        let elements = ctx.arena.alloc_from_iter(arguments.iter().map(|arg| {
            let arg_expr = ctx.lower_expr(&arg.expr);
            ctx.expr(
                arg.expr.span.with_ctxt(macsp.ctxt()),
                hir::ExprKind::AddrOf(hir::BorrowKind::Ref, hir::Mutability::Not, arg_expr),
            )
        }));
        let args_tuple = ctx.arena.alloc(ctx.expr(macsp, hir::ExprKind::Tup(elements)));
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
    struct MayContainYieldPoint;

    impl Visitor<'_> for MayContainYieldPoint {
        type Result = ControlFlow<()>;

        fn visit_expr(&mut self, e: &ast::Expr) -> ControlFlow<()> {
            if let ast::ExprKind::Await(_, _) | ast::ExprKind::Yield(_) = e.kind {
                ControlFlow::Break(())
            } else {
                visit::walk_expr(self, e)
            }
        }

        fn visit_mac_call(&mut self, _: &ast::MacCall) -> ControlFlow<()> {
            // Macros should be expanded at this point.
            unreachable!("unexpanded macro in ast lowering");
        }

        fn visit_item(&mut self, _: &ast::Item) -> ControlFlow<()> {
            // Do not recurse into nested items.
            ControlFlow::Continue(())
        }
    }

    MayContainYieldPoint.visit_expr(e).is_break()
}

fn for_all_argument_indexes(template: &mut [FormatArgsPiece], mut f: impl FnMut(&mut usize)) {
    for piece in template {
        let FormatArgsPiece::Placeholder(placeholder) = piece else { continue };
        if let Ok(index) = &mut placeholder.argument.index {
            f(index);
        }
        if let Some(FormatCount::Argument(FormatArgPosition { index: Ok(index), .. })) =
            &mut placeholder.format_options.width
        {
            f(index);
        }
        if let Some(FormatCount::Argument(FormatArgPosition { index: Ok(index), .. })) =
            &mut placeholder.format_options.precision
        {
            f(index);
        }
    }
}
