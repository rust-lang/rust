use std::borrow::Cow;

use rustc_ast::*;
use rustc_data_structures::fx::FxIndexMap;
use rustc_hir as hir;
use rustc_session::config::FmtDebug;
use rustc_span::{ByteSymbol, DesugaringKind, Ident, Span, Symbol, sym};

use super::LoweringContext;

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

        for i in 0..fmt.template.len() {
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

/// Get the value for a `width` or `precision` field.
///
/// Returns the value and whether it is indirect (an indexed argument) or not.
fn make_count(
    count: &FormatCount,
    argmap: &mut FxIndexMap<(usize, ArgumentType), Option<Span>>,
) -> (bool, u16) {
    match count {
        FormatCount::Literal(n) => (false, *n),
        FormatCount::Argument(arg) => (
            true,
            argmap.insert_full((arg.index.unwrap_or(usize::MAX), ArgumentType::Usize), arg.span).0
                as u16,
        ),
    }
}

fn expand_format_args<'hir>(
    ctx: &mut LoweringContext<'_, 'hir>,
    macsp: Span,
    fmt: &FormatArgs,
    allow_const: bool,
) -> hir::ExprKind<'hir> {
    let macsp = ctx.lower_span(macsp);

    // Create a list of all _unique_ (argument, format trait) combinations.
    // E.g. "{0} {0:x} {0} {1}" -> [(0, Display), (0, LowerHex), (1, Display)]
    //
    // We use usize::MAX for arguments that don't exist, because that can never be a valid index
    // into the arguments array.
    let mut argmap = FxIndexMap::default();

    let mut incomplete_lit = String::new();

    let mut implicit_arg_index = 0;

    let mut bytecode = Vec::new();

    let template = if fmt.template.is_empty() {
        // Treat empty templates as a single literal piece (with an empty string),
        // so we produce `from_str("")` for those.
        &[FormatArgsPiece::Literal(sym::empty)][..]
    } else {
        &fmt.template[..]
    };

    // See library/core/src/fmt/mod.rs for the format string encoding format.

    for (i, piece) in template.iter().enumerate() {
        match piece {
            &FormatArgsPiece::Literal(sym) => {
                // Coalesce adjacent literal pieces.
                if let Some(FormatArgsPiece::Literal(_)) = template.get(i + 1) {
                    incomplete_lit.push_str(sym.as_str());
                    continue;
                }
                let mut s = if incomplete_lit.is_empty() {
                    sym.as_str()
                } else {
                    incomplete_lit.push_str(sym.as_str());
                    &incomplete_lit
                };

                // If this is the last piece and was the only piece, that means
                // there are no placeholders and the entire format string is just a literal.
                //
                // In that case, we can just use `from_str`.
                if i + 1 == template.len() && bytecode.is_empty() {
                    // Generate:
                    //     <core::fmt::Arguments>::from_str("meow")
                    let from_str = ctx.arena.alloc(ctx.expr_lang_item_type_relative(
                        macsp,
                        hir::LangItem::FormatArguments,
                        if allow_const { sym::from_str } else { sym::from_str_nonconst },
                    ));
                    let sym = if incomplete_lit.is_empty() { sym } else { Symbol::intern(s) };
                    let s = ctx.expr_str(fmt.span, sym);
                    let args = ctx.arena.alloc_from_iter([s]);
                    return hir::ExprKind::Call(from_str, args);
                }

                // Encode the literal in chunks of up to u16::MAX bytes, split at utf-8 boundaries.
                while !s.is_empty() {
                    let len = s.floor_char_boundary(usize::from(u16::MAX));
                    if len < 0x80 {
                        bytecode.push(len as u8);
                    } else {
                        bytecode.push(0x80);
                        bytecode.extend_from_slice(&(len as u16).to_le_bytes());
                    }
                    bytecode.extend(&s.as_bytes()[..len]);
                    s = &s[len..];
                }

                incomplete_lit.clear();
            }
            FormatArgsPiece::Placeholder(p) => {
                // Push the start byte and remember its index so we can set the option bits later.
                let i = bytecode.len();
                bytecode.push(0xC0);

                let position = argmap
                    .insert_full(
                        (
                            p.argument.index.unwrap_or(usize::MAX),
                            ArgumentType::Format(p.format_trait),
                        ),
                        p.span,
                    )
                    .0 as u64;

                // This needs to match the constants in library/core/src/fmt/mod.rs.
                let o = &p.format_options;
                let align = match o.alignment {
                    Some(FormatAlignment::Left) => 0,
                    Some(FormatAlignment::Right) => 1,
                    Some(FormatAlignment::Center) => 2,
                    None => 3,
                };
                let default_flags = 0x6000_0020;
                let flags: u32 = o.fill.unwrap_or(' ') as u32
                    | ((o.sign == Some(FormatSign::Plus)) as u32) << 21
                    | ((o.sign == Some(FormatSign::Minus)) as u32) << 22
                    | (o.alternate as u32) << 23
                    | (o.zero_pad as u32) << 24
                    | ((o.debug_hex == Some(FormatDebugHex::Lower)) as u32) << 25
                    | ((o.debug_hex == Some(FormatDebugHex::Upper)) as u32) << 26
                    | (o.width.is_some() as u32) << 27
                    | (o.precision.is_some() as u32) << 28
                    | align << 29;
                if flags != default_flags {
                    bytecode[i] |= 1;
                    bytecode.extend_from_slice(&flags.to_le_bytes());
                    if let Some(val) = &o.width {
                        let (indirect, val) = make_count(val, &mut argmap);
                        // Only encode if nonzero; zero is the default.
                        if indirect || val != 0 {
                            bytecode[i] |= 1 << 1 | (indirect as u8) << 4;
                            bytecode.extend_from_slice(&val.to_le_bytes());
                        }
                    }
                    if let Some(val) = &o.precision {
                        let (indirect, val) = make_count(val, &mut argmap);
                        // Only encode if nonzero; zero is the default.
                        if indirect || val != 0 {
                            bytecode[i] |= 1 << 2 | (indirect as u8) << 5;
                            bytecode.extend_from_slice(&val.to_le_bytes());
                        }
                    }
                }
                if implicit_arg_index != position {
                    bytecode[i] |= 1 << 3;
                    bytecode.extend_from_slice(&(position as u16).to_le_bytes());
                }
                implicit_arg_index = position + 1;
            }
        }
    }

    assert!(incomplete_lit.is_empty());

    // Zero terminator.
    bytecode.push(0);

    // Ensure all argument indexes actually fit in 16 bits, as we truncated them to 16 bits before.
    if argmap.len() > u16::MAX as usize {
        ctx.dcx().span_err(macsp, "too many format arguments");
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
                if let Some(arg) = arguments.get(arg_index) {
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
    //     unsafe {
    //         <core::fmt::Arguments>::new(b"…", &args)
    //     }
    let template = ctx.expr_byte_str(macsp, ByteSymbol::intern(&bytecode));
    let call = {
        let new = ctx.arena.alloc(ctx.expr_lang_item_type_relative(
            macsp,
            hir::LangItem::FormatArguments,
            sym::new,
        ));
        let args = ctx.expr_ref(macsp, args);
        let new_args = ctx.arena.alloc_from_iter([template, args]);
        ctx.expr_call(macsp, new, new_args)
    };
    let call = hir::ExprKind::Block(
        ctx.arena.alloc(hir::Block {
            stmts: &[],
            expr: Some(call),
            hir_id: ctx.next_id(),
            rules: hir::BlockCheckMode::UnsafeBlock(hir::UnsafeSource::CompilerGenerated),
            span: macsp,
            targeted_by_break: false,
        }),
        None,
    );

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
