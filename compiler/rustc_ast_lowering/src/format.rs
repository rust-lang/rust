use super::LoweringContext;
use hir::def::{DefKind, Res};
use hir::definitions::DefPathData;
use rustc_ast as ast;
use rustc_ast::visit::{self, Visitor};
use rustc_ast::*;
use rustc_data_structures::fx::FxIndexMap;
use rustc_hir as hir;
use rustc_session::Session;
use rustc_span::{
    def_id::LocalDefId,
    sym,
    symbol::{kw, Ident},
    Span, Symbol,
};
use rustc_target::spec::abi::Abi;
use std::borrow::Cow;

impl<'hir> LoweringContext<'_, 'hir> {
    pub(crate) fn lower_format_args(&mut self, sp: Span, fmt: &FormatArgs) -> hir::ExprKind<'hir> {
        // Optimize the format arguments
        let (allow_const, fmt) = process_args(self.tcx.sess, fmt);

        // Generate access of the `index` argument.
        let arg_access = |this: &mut LoweringContext<'_, 'hir>, index: usize, span| {
            let arg = this.lower_expr(&fmt.arguments.all_args()[index].expr);
            this.expr(span, hir::ExprKind::AddrOf(hir::BorrowKind::Ref, hir::Mutability::Not, arg))
        };

        match fmt.panic {
            ast::FormatPanicKind::Format => {
                expand_format_args(self, sp, &fmt, allow_const, arg_access)
            }
            ast::FormatPanicKind::Panic { id, constness: _ } => {
                self.lower_panic_args(id, sp, &fmt, arg_access)
            }
        }
    }

    fn lower_panic_args(
        &mut self,
        cold_path: NodeId,
        span: Span,
        fmt: &FormatArgs,
        mut arg_access: impl FnMut(&mut LoweringContext<'_, 'hir>, usize, Span) -> hir::Expr<'hir>,
    ) -> hir::ExprKind<'hir> {
        // Call the cold path function passing on the arguments required for formatting.
        let span = self.lower_span(span);
        let arena = self.arena;

        let args = &*arena.alloc_from_iter(
            fmt.arguments
                .all_args()
                .iter()
                .enumerate()
                .map(|(i, arg)| arg_access(self, i, arg.expr.span.with_ctxt(span.ctxt()))),
        );

        let item = self.local_def_id(cold_path);
        let res = Res::Def(DefKind::Fn, item.to_def_id());
        let path_hir_id = self.next_id();
        let path = hir::ExprKind::Path(hir::QPath::Resolved(
            None,
            arena.alloc(hir::Path {
                span,
                res,
                segments: arena_vec![self; hir::PathSegment::new(
                    Ident::new(sym::panic_cold, span), path_hir_id, res
                )],
            }),
        ));
        let path = self.expr(span, path);
        let call = arena.alloc(self.expr(span, hir::ExprKind::Call(arena.alloc(path), args)));
        let item = self.stmt(
            span,
            hir::StmtKind::Item(hir::ItemId { owner_id: hir::OwnerId { def_id: item } }),
        );
        let stmts = arena_vec![self; item];
        hir::ExprKind::Block(self.block_all(span, stmts, Some(call)), None)
    }

    fn generic_ty(&mut self, span: Span, param: &hir::GenericParam<'_>) -> &'hir hir::Ty<'hir> {
        let arena = self.arena;
        let path_hir_id = self.next_id();
        let res = Res::Def(DefKind::TyParam, param.def_id.to_def_id());
        let path = hir::QPath::Resolved(
            None,
            arena.alloc(hir::Path {
                span,
                res,
                segments: arena_vec![self; hir::PathSegment::new(param.name.ident(), path_hir_id, res)],
            }),
        );
        arena.alloc(self.ty(span, hir::TyKind::Path(path)))
    }

    fn create_generic_param(
        &mut self,
        span: Span,
        name: Symbol,
        data: DefPathData,
        kind: hir::GenericParamKind<'hir>,
    ) -> hir::GenericParam<'hir> {
        let node_id = self.next_node_id();
        let def_id = self.create_def(self.current_hir_id_owner.def_id, node_id, data, span);
        let hir_id = self.lower_node_id(node_id);
        hir::GenericParam {
            def_id,
            hir_id,
            name: hir::ParamName::Plain(Ident { name, span }),
            span,
            kind,
            colon_span: None,
            pure_wrt_drop: false,
            source: hir::GenericParamSource::Generics,
        }
    }

    fn generic_bounds(
        &mut self,
        span: Span,
        traits: &[FormatTrait],
    ) -> &'hir [hir::GenericBound<'hir>] {
        // Maps from the format trait to the lang item
        let map_trait = |t| match t {
            FormatTrait::Display => hir::LangItem::FormatDisplay,
            FormatTrait::Debug => hir::LangItem::FormatDebug,
            FormatTrait::LowerExp => hir::LangItem::FormatLowerExp,
            FormatTrait::UpperExp => hir::LangItem::FormatUpperExp,
            FormatTrait::Octal => hir::LangItem::FormatOctal,
            FormatTrait::Pointer => hir::LangItem::FormatPointer,
            FormatTrait::Binary => hir::LangItem::FormatBinary,
            FormatTrait::LowerHex => hir::LangItem::FormatLowerHex,
            FormatTrait::UpperHex => hir::LangItem::FormatUpperHex,
        };

        self.arena.alloc_from_iter(traits.iter().map(|t| {
            hir::GenericBound::LangItemTrait(
                map_trait(*t),
                span,
                self.next_id(),
                self.arena.alloc(hir::GenericArgs::none()),
            )
        }))
    }

    /// Lowers the cold path function for panic_args!
    pub(crate) fn lower_panic_args_cold(
        &mut self,
        fmt: &FormatArgs,
        span: Span,
    ) -> &'hir hir::Item<'hir> {
        let FormatPanicKind::Panic { id, constness } = fmt.panic else { panic!() };

        let (allow_const, fmt) = process_args(self.tcx.sess, fmt);

        let arena = self.arena;

        let span = self.lower_span(span);
        let hir_id = self.lower_node_id(id);

        let arg_count = fmt.arguments.all_args().len();

        // Create the generic parameters `A0`, `A1`, .., `Ai`.
        let mut generics: Vec<_> = (0..arg_count)
            .map(|i| {
                let name = Symbol::intern(&format!("A{i}"));
                self.create_generic_param(
                    span,
                    name,
                    DefPathData::TypeNs(name),
                    hir::GenericParamKind::Type { default: None, synthetic: false },
                )
            })
            .collect();

        // Create the lifetime 'a used in the parameter types.
        let lifetime_name = Symbol::intern("'a");
        let lifetime: LocalDefId = {
            let param = self.create_generic_param(
                span,
                lifetime_name,
                DefPathData::LifetimeNs(lifetime_name),
                hir::GenericParamKind::Lifetime { kind: hir::LifetimeParamKind::Explicit },
            );
            let def_id = param.def_id;
            generics.insert(0, param);
            def_id
        };

        let body_id = self.lower_body(|this| {
            // Create parameter bindings `a0`, `a1`, .., `ai`.
            let bindings: Vec<(hir::HirId, Ident)> = (0..arg_count)
                .map(|i| (this.next_id(), Ident::new(Symbol::intern(&format!("a{i}")), span)))
                .collect();
            let params = arena.alloc_from_iter((0..arg_count).map(|i| {
                let pat = arena.alloc(hir::Pat {
                    hir_id: bindings[i].0,
                    kind: hir::PatKind::Binding(
                        BindingAnnotation::NONE,
                        bindings[i].0,
                        bindings[i].1,
                        None,
                    ),
                    span,
                    default_binding_modes: true,
                });
                let hir_id = this.next_id();
                hir::Param { hir_id, pat, ty_span: span, span }
            }));

            // Generate access of the `i` format argument via parameter binding `ai`.
            let arg_access = |this: &mut LoweringContext<'_, 'hir>, i: usize, span| {
                this.expr_ident_mut(span, bindings[i].1, bindings[i].0)
            };

            // Generate formatting code.
            let args = expand_format_args(this, span, &fmt, allow_const, arg_access);
            let args = this.expr(span, args);

            // Call `panic_fmt` with the generated `Arguments`.
            let panic = arena.alloc(this.expr_call_lang_item_fn_mut(
                span,
                hir::LangItem::PanicFmt,
                arena_vec![this; args],
                None,
            ));

            let block = arena.alloc(hir::Block {
                stmts: &[],
                expr: Some(panic),
                hir_id: this.next_id(),
                rules: hir::BlockCheckMode::DefaultBlock,
                span,
                targeted_by_break: false,
            });
            (params, this.expr_block(block))
        });

        // Compute trait bounds we need to apply to each format argument.
        let mut arg_traits: Vec<Vec<_>> = (0..arg_count).map(|_| Vec::new()).collect();
        for piece in &fmt.template {
            let FormatArgsPiece::Placeholder(placeholder) = piece else { continue };
            if let Ok(index) = placeholder.argument.index {
                if !arg_traits[index].iter().any(|t| *t == placeholder.format_trait) {
                    arg_traits[index].push(placeholder.format_trait);
                }
            }
        }

        // Create where bound required for format arguments, like, A0: Display + Debug.
        let predicates =
            arena.alloc_from_iter(arg_traits.into_iter().enumerate().filter_map(|(i, traits)| {
                (!traits.is_empty()).then(|| {
                    hir::WherePredicate::BoundPredicate(hir::WhereBoundPredicate {
                        hir_id: self.next_id(),
                        span,
                        origin: hir::PredicateOrigin::GenericParam,
                        bound_generic_params: &[],
                        bounded_ty: self.generic_ty(span, &generics[1 + i]),
                        bounds: self.generic_bounds(span, &traits),
                    })
                })
            }));

        // Create input parameter types &'a A0, &'a A1, .., &'a Ai
        let inputs = arena.alloc_from_iter((0..arg_count).map(|i| {
            let ty = self.generic_ty(span, &generics[1 + i]);
            let hir_id = self.next_id();
            let lifetime = arena.alloc(hir::Lifetime {
                hir_id,
                ident: Ident::new(lifetime_name, span),
                res: hir::LifetimeName::Param(lifetime),
            });
            self.ty(span, hir::TyKind::Ref(lifetime, hir::MutTy { ty, mutbl: Mutability::Not }))
        }));

        let decl = arena.alloc(hir::FnDecl {
            inputs,
            // Return type !
            output: hir::FnRetTy::Return(self.arena.alloc(hir::Ty {
                kind: hir::TyKind::Never,
                span,
                hir_id: self.next_id(),
            })),
            c_variadic: false,
            lifetime_elision_allowed: false,
            implicit_self: hir::ImplicitSelfKind::None,
        });
        let sig = hir::FnSig {
            decl,
            header: hir::FnHeader {
                unsafety: hir::Unsafety::Normal,
                asyncness: hir::IsAsync::NotAsync,
                constness: self.lower_constness(constness),
                abi: Abi::Rust,
            },
            span,
        };
        let generics = arena.alloc(hir::Generics {
            params: arena.alloc_from_iter(generics),
            predicates,
            has_where_clause_predicates: false,
            where_clause_span: span,
            span,
        });
        let kind = hir::ItemKind::Fn(sig, generics, body_id);
        let g = &self.tcx.sess.parse_sess.attr_id_generator;
        self.lower_attrs(
            hir_id,
            &[
                attr::mk_attr_nested_word(g, ast::AttrStyle::Outer, sym::inline, sym::never, span),
                attr::mk_attr_word(g, ast::AttrStyle::Outer, sym::track_caller, span),
                attr::mk_attr_word(g, ast::AttrStyle::Outer, sym::cold, span),
            ],
        );
        arena.alloc(hir::Item {
            owner_id: hir_id.expect_owner(),
            ident: Ident::new(sym::panic_cold, span),
            kind,
            vis_span: span,
            span,
        })
    }
}

fn process_args<'a>(sess: &Session, fmt: &'a FormatArgs) -> (bool, Cow<'a, FormatArgs>) {
    // Never call the const constructor of `fmt::Arguments` if the
    // format_args!() had any arguments _before_ flattening/inlining.
    let allow_const = fmt.arguments.all_args().is_empty();
    let mut fmt = Cow::Borrowed(fmt);
    if sess.opts.unstable_opts.flatten_format_args {
        fmt = flatten_format_args(fmt);
        fmt = inline_literals(fmt);
    }
    (allow_const, fmt)
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
            let fmt2 = loop { // Unwrap the Expr to get to the FormatArgs.
                match &mut fmt2.kind {
                    ExprKind::Paren(inner) | ExprKind::AddrOf(BorrowKind::Ref, _, inner) => fmt2 = inner,
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

/// Inline literals into the format string.
///
/// Turns
///
/// `format_args!("Hello, {}! {} {}", "World", 123, x)`
///
/// into
///
/// `format_args!("Hello, World! 123 {}", x)`.
fn inline_literals(mut fmt: Cow<'_, FormatArgs>) -> Cow<'_, FormatArgs> {
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
            if let token::LitKind::Str | token::LitKind::StrRaw(_) = lit.kind
                && let Ok(LitKind::Str(s, _)) = LitKind::from_token_lit(lit)
            {
                literal = Some(s);
            } else if let token::LitKind::Integer = lit.kind
                && let Ok(LitKind::Int(n, _)) = LitKind::from_token_lit(lit)
            {
                literal = Some(Symbol::intern(&n.to_string()));
            }
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
            let value = ctx.arena.alloc_from_iter([ctx.expr_usize(sp, *n)]);
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
                        ctx.tcx.sess.delay_span_bug(sp, "lowered bad format_args count"),
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
///     <core::fmt::rt::Placeholder::new(
///         …usize, // position
///         '…', // fill
///         <core::fmt::rt::Alignment>::…, // alignment
///         …u32, // flags
///         <core::fmt::rt::Count::…>, // width
///         <core::fmt::rt::Count::…>, // precision
///     )
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
            hir::ExprKind::Err(ctx.tcx.sess.delay_span_bug(sp, "lowered bad format_args count")),
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
    // This needs to match `Flag` in library/core/src/fmt/rt.rs.
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
    allow_const: bool,
    mut arg_access: impl FnMut(&mut LoweringContext<'_, 'hir>, usize, Span) -> hir::Expr<'hir>,
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
        let elements = ctx.arena.alloc_from_iter(arguments.iter().enumerate().zip(argmap).map(
            |((i, arg), ((_, ty), placeholder_span))| {
                let placeholder_span =
                    placeholder_span.unwrap_or(arg.expr.span).with_ctxt(macsp.ctxt());
                let arg_span = match arg.kind {
                    FormatArgumentKind::Captured(_) => placeholder_span,
                    _ => arg.expr.span.with_ctxt(macsp.ctxt()),
                };
                let ref_arg = ctx.arena.alloc(arg_access(ctx, i, arg_span));
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
        let elements = ctx.arena.alloc_from_iter(
            arguments
                .iter()
                .enumerate()
                .map(|(i, arg)| arg_access(ctx, i, arg.expr.span.with_ctxt(macsp.ctxt()))),
        );
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
    struct MayContainYieldPoint(bool);

    impl Visitor<'_> for MayContainYieldPoint {
        fn visit_expr(&mut self, e: &ast::Expr) {
            if let ast::ExprKind::Await(_, _) | ast::ExprKind::Yield(_) = e.kind {
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
