use std::borrow::Cow;

use rustc_ast::*;
use rustc_data_structures::fx::FxIndexMap;
use rustc_hir as hir;
use rustc_session::config::FmtDebug;
use rustc_span::{DesugaringKind, Ident, Span, Symbol, sym};

use super::LoweringContext;
use super::errors::TooManyFormatArguments;

impl<'hir> LoweringContext<'_, 'hir> {
    pub(crate) fn lower_format_args(&mut self, sp: Span, fmt: &FormatArgs) -> hir::ExprKind<'hir> {
        // Never call the const constructor of `fmt::Arguments` if the
        // format_args!() had any arguments _before_ flattening/inlining.
        let allow_const = fmt.arguments.all_args().is_empty();
        let mut fmt = Cow::Borrowed(fmt);

        let sp = self.mark_span_with_reason(
            DesugaringKind::FormatLiteral { source: fmt.is_source_literal },
            sp,
            sp.ctxt().outer_expn_data().allow_internal_unstable,
        );

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
            IntTy::Isize => self.tcx.data_layout.pointer_size().signed_int_max() as u128,
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
            UintTy::Usize => self.tcx.data_layout.pointer_size().unsigned_int_max(),
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

        let mut i = 0;

        while i < fmt.template.len() {
            if let FormatArgsPiece::Placeholder(placeholder) = &fmt.template[i]
                && let Ok(arg_index) = placeholder.argument.index
                && let FormatTrait::Display = placeholder.format_trait
                && placeholder.format_options == Default::default()
                && let arg = fmt.arguments.all_args()[arg_index].expr.peel_parens_and_refs()
                && let ExprKind::Lit(lit) = arg.kind
                && let Some(literal) = self.try_inline_lit(lit)
            {
                // Now we need to mutate the outer FormatArgs.
                // If this is the first time, this clones the outer FormatArgs.
                let fmt = fmt.to_mut();
                // Replace the placeholder with the literal.
                if literal.is_empty() {
                    fmt.template.remove(i);
                } else {
                    fmt.template[i] = FormatArgsPiece::Literal(literal);
                    i += 1;
                }
                was_inlined[arg_index] = true;
                inlined_anything = true;
            } else {
                i += 1;
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
    Constant(u16),
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
            Usize | Constant(_) => sym::from_usize,
        },
    ));
    ctx.expr_call_mut(sp, new_fn, std::slice::from_ref(arg))
}

/// Generate a hir expression for a format_piece.
///
/// Generates:
///
/// ```text
///     <core::fmt::rt::Piece>::…(…)
/// ```
fn make_piece<'hir>(
    ctx: &mut LoweringContext<'_, 'hir>,
    constructor: Symbol,
    expr: hir::Expr<'hir>,
    sp: Span,
) -> hir::Expr<'hir> {
    let new_fn = ctx.arena.alloc(ctx.expr_lang_item_type_relative(
        sp,
        hir::LangItem::FormatPiece,
        constructor,
    ));
    let new_args = ctx.arena.alloc_from_iter([expr]);
    ctx.expr(sp, hir::ExprKind::Call(new_fn, new_args))
}

/// Generate the 64 bit descriptor for a format_args placeholder specification.
fn make_format_spec(
    placeholder: &FormatPlaceholder,
    argmap: &mut FxIndexMap<(usize, ArgumentType), Option<Span>>,
) -> u64 {
    let position = argmap
        .insert_full(
            (
                placeholder.argument.index.unwrap_or(usize::MAX),
                ArgumentType::Format(placeholder.format_trait),
            ),
            placeholder.span,
        )
        .0 as u64;
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
    let (width_indirect, width) = make_count(width, argmap);
    let (precision_indirect, precision) = make_count(precision, argmap);
    (flags as u64) << 32
        | (precision_indirect as u64) << 31
        | (width_indirect as u64) << 30
        | precision << 20
        | width << 10
        | position
}

fn make_count(
    count: &Option<FormatCount>,
    argmap: &mut FxIndexMap<(usize, ArgumentType), Option<Span>>,
) -> (bool, u64) {
    match count {
        None => (false, 0),
        &Some(FormatCount::Literal(n)) => {
            if n < 1 << 10 {
                (false, n as u64)
            } else {
                // Too big. Upgrade to an argument.
                let index =
                    argmap.insert_full((usize::MAX, ArgumentType::Constant(n)), None).0 as u64;
                (true, index)
            }
        }
        Some(FormatCount::Argument(arg)) => (
            true,
            argmap.insert_full((arg.index.unwrap_or(usize::MAX), ArgumentType::Usize), arg.span).0
                as u64,
        ),
    }
}

fn expand_format_args<'hir>(
    ctx: &mut LoweringContext<'_, 'hir>,
    macsp: Span,
    fmt: &FormatArgs,
    allow_const: bool,
) -> hir::ExprKind<'hir> {
    // Create a list of all _unique_ (argument, format trait) combinations.
    // E.g. "{0} {0:x} {0} {1}" -> [(0, Display), (0, LowerHex), (1, Display)]
    //
    // We use usize::MAX for arguments that don't exist, because that can never be a valid index
    // into the arguments array.
    let mut argmap = FxIndexMap::default();

    // Generate:
    //
    // ```
    // &[
    //      Piece::num(4),
    //      Piece::str("meow"),
    //      Piece::num(0xE000_0020_0000_0002),
    //      …,
    //      Piece::num(0),
    // ]
    // ```
    let mut pieces = Vec::new();

    let mut incomplete_lit = String::new();

    let default_options = 0xE000_0020_0000_0000;
    let mut implicit_arg_index = 0;

    let template = if fmt.template.is_empty() {
        // Treat empty templates as a single literal piece (with an empty string),
        // so we produce `from_str("")` for those.
        &[FormatArgsPiece::Literal(sym::empty)][..]
    } else {
        &fmt.template[..]
    };

    for (i, piece) in template.iter().enumerate() {
        match piece {
            &FormatArgsPiece::Literal(sym) => {
                // Coalesce adjacent literal pieces.
                if let Some(FormatArgsPiece::Literal(_)) = template.get(i + 1) {
                    incomplete_lit.push_str(sym.as_str());
                    continue;
                }
                let (sym, len) = if incomplete_lit.is_empty() {
                    (sym, sym.as_str().len())
                } else {
                    incomplete_lit.push_str(sym.as_str());
                    let sym = Symbol::intern(&incomplete_lit);
                    let len = incomplete_lit.len();
                    incomplete_lit.clear();
                    (sym, len)
                };

                // If this is the last piece and was the only piece, that means
                // there are no placeholders and the entire format string is just a literal.
                //
                // In that case, we don't need an array of `Piece`s: we can just use `from_str`.
                if i + 1 == template.len() && pieces.is_empty() {
                    // Generate:
                    //     <core::fmt::Arguments>::from_str("meow")
                    let from_str = ctx.arena.alloc(ctx.expr_lang_item_type_relative(
                        macsp,
                        hir::LangItem::FormatArguments,
                        if allow_const { sym::from_str } else { sym::from_str_nonconst },
                    ));
                    let s = ctx.expr_str(fmt.span, sym);
                    let args = ctx.arena.alloc_from_iter([s]);
                    return hir::ExprKind::Call(from_str, args);
                }

                // Producing a `Piece::num(0)` would be problematic, as that is the terminator.
                assert!(len > 0);

                // ```
                //  Piece::num(4),
                // ```
                let i = ctx.expr_usize(macsp, len as u64);
                pieces.push(make_piece(ctx, sym::num, i, macsp));

                // ```
                //  Piece::str("meow"),
                // ```
                let s = ctx.expr_str(fmt.span, sym);
                pieces.push(make_piece(ctx, sym::str, s, macsp));
            }
            FormatArgsPiece::Placeholder(p) => {
                // ```
                //  Piece::num(0xE000_0020_0000_0000),
                // ```
                // Or, on 32 bit platforms:
                // ```
                //  Piece::num(0xE000_0020),
                //  Piece::num(0x0000_0000),
                // ```
                // Or, on 16 bit platforms:
                // ```
                //  Piece::num(0xE000),
                //  Piece::num(0x0020),
                //  Piece::num(0x0000),
                //  Piece::num(0x0000),
                // ```

                let bits = make_format_spec(p, &mut argmap);

                // If this placeholder uses the next argument index, is surrounded by literal string
                // pieces, and uses default formatting options, then we can skip it, as this kind of
                // placeholder is implied by two consecutive string pieces.
                if bits == default_options + implicit_arg_index {
                    if let (Some(FormatArgsPiece::Literal(_)), Some(FormatArgsPiece::Literal(_))) =
                        (template.get(i.wrapping_sub(1)), template.get(i + 1))
                    {
                        implicit_arg_index += 1;
                        continue;
                    }
                }

                if ctx.tcx.sess.target.pointer_width >= 64 {
                    let bits = ctx.expr_usize(macsp, bits);
                    pieces.push(make_piece(ctx, sym::num, bits, macsp));
                } else if ctx.tcx.sess.target.pointer_width >= 32 {
                    let high = ctx.expr_usize(macsp, bits >> 32);
                    let low = ctx.expr_usize(macsp, bits & 0xFFFF_FFFF);
                    pieces.push(make_piece(ctx, sym::num, high, macsp));
                    pieces.push(make_piece(ctx, sym::num, low, macsp));
                } else {
                    let w1 = ctx.expr_usize(macsp, bits >> 48);
                    let w2 = ctx.expr_usize(macsp, bits >> 32 & 0xFFFF);
                    let w3 = ctx.expr_usize(macsp, bits >> 16 & 0xFFFF);
                    let w4 = ctx.expr_usize(macsp, bits & 0xFFFF);
                    pieces.push(make_piece(ctx, sym::num, w1, macsp));
                    pieces.push(make_piece(ctx, sym::num, w2, macsp));
                    pieces.push(make_piece(ctx, sym::num, w3, macsp));
                    pieces.push(make_piece(ctx, sym::num, w4, macsp));
                }

                implicit_arg_index = (bits & 0x3FF) + 1;
            }
        }
    }

    assert!(incomplete_lit.is_empty());

    // Zero terminator.
    //
    // ```
    //  Piece::num(0),
    // ```
    let zero = ctx.expr_usize(macsp, 0);
    pieces.push(make_piece(ctx, sym::num, zero, macsp));

    // ```
    //   unsafe { <core::fmt::rt::Template>::new(const { &[pieces…] }) }
    // ```
    let template_new =
        ctx.expr_lang_item_type_relative(macsp, hir::LangItem::FormatTemplate, sym::new);
    let pieces = ctx.expr_array_ref(macsp, ctx.arena.alloc_from_iter(pieces));
    let pieces = ctx.expr_const(macsp, pieces);
    let template = ctx.expr(
        macsp,
        hir::ExprKind::Call(ctx.arena.alloc(template_new), ctx.arena.alloc_from_iter([pieces])),
    );
    let template = ctx.expr_unsafe(macsp, ctx.arena.alloc(template));

    // Ensure all argument indexes actually fit in 10 bits, as we truncated them to 10 bits before.
    if argmap.len() >= 1 << 10 {
        ctx.dcx().emit_err(TooManyFormatArguments { span: fmt.span });
    }

    let arguments = fmt.arguments.all_args();

    let (let_statements, args) = if arguments.is_empty() {
        // Generate:
        //     []
        (vec![], ctx.arena.alloc(ctx.expr(macsp, hir::ExprKind::Array(&[]))))
    } else {
        // Generate:
        //     super let args = (&arg0, &arg1, &…);
        let args_ident = Ident::new(sym::args, macsp);
        let (args_pat, args_hir_id) = ctx.pat_ident(macsp, args_ident);
        let elements = ctx.arena.alloc_from_iter(arguments.iter().map(|arg| {
            let arg_expr = ctx.lower_expr(&arg.expr);
            ctx.expr(
                arg.expr.span.with_ctxt(macsp.ctxt()),
                hir::ExprKind::AddrOf(hir::BorrowKind::Ref, hir::Mutability::Not, arg_expr),
            )
        }));
        let args_tuple = ctx.arena.alloc(ctx.expr(macsp, hir::ExprKind::Tup(elements)));
        let let_statement_1 = ctx.stmt_super_let_pat(macsp, args_pat, Some(args_tuple));

        // Generate:
        //     super let args = [
        //         <core::fmt::Argument>::new_display(args.0),
        //         <core::fmt::Argument>::new_lower_hex(args.1),
        //         <core::fmt::Argument>::new_debug(args.0),
        //         …
        //     ];
        let args = ctx.arena.alloc_from_iter(argmap.iter().map(
            |(&(arg_index, ty), &placeholder_span)| {
                if let ArgumentType::Constant(c) = ty {
                    let arg = ctx.arena.alloc(ctx.expr_usize(macsp, c.into()));
                    let arg = ctx.arena.alloc(ctx.expr(
                        macsp,
                        hir::ExprKind::AddrOf(hir::BorrowKind::Ref, hir::Mutability::Not, arg),
                    ));
                    make_argument(ctx, macsp, arg, ty)
                } else if let Some(arg) = arguments.get(arg_index) {
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
                } else {
                    ctx.expr(
                        macsp,
                        hir::ExprKind::Err(
                            ctx.dcx().span_delayed_bug(macsp, "missing format_args argument"),
                        ),
                    )
                }
            },
        ));
        let args = ctx.arena.alloc(ctx.expr(macsp, hir::ExprKind::Array(args)));
        let (args_pat, args_hir_id) = ctx.pat_ident(macsp, args_ident);
        let let_statement_2 = ctx.stmt_super_let_pat(macsp, args_pat, Some(args));
        (
            vec![let_statement_1, let_statement_2],
            ctx.arena.alloc(ctx.expr_ident_mut(macsp, args_ident, args_hir_id)),
        )
    };

    // Generate:
    //     <core::fmt::Arguments>::new(
    //         template,
    //         &args,
    //     )
    let call = {
        let new = ctx.arena.alloc(ctx.expr_lang_item_type_relative(
            macsp,
            hir::LangItem::FormatArguments,
            sym::new,
        ));
        let args = ctx.expr_ref(macsp, args);
        let new_args = ctx.arena.alloc_from_iter([template, args]);
        hir::ExprKind::Call(new, new_args)
    };

    if !let_statements.is_empty() {
        // Generate:
        //     {
        //         super let …
        //         super let …
        //         <core::fmt::Arguments>::new(…)
        //     }
        let call = ctx.arena.alloc(ctx.expr(macsp, call));
        let block = ctx.block_all(macsp, ctx.arena.alloc_from_iter(let_statements), Some(call));
        hir::ExprKind::Block(block, None)
    } else {
        call
    }
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
