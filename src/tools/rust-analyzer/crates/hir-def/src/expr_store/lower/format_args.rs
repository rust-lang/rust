//! Lowering of `format_args!()`.

use base_db::FxIndexSet;
use hir_expand::name::Name;
use intern::{Symbol, sym};
use span::SyntaxContext;
use syntax::{AstPtr, AstToken as _, ast};

use crate::{
    builtin_type::BuiltinUint,
    expr_store::{HygieneId, lower::ExprCollector, path::Path},
    hir::{
        Array, BindingAnnotation, Expr, ExprId, Literal, Pat, RecordLitField, RecordSpread,
        Statement,
        format_args::{
            self, FormatAlignment, FormatArgs, FormatArgsPiece, FormatArgument, FormatArgumentKind,
            FormatArgumentsCollector, FormatCount, FormatDebugHex, FormatOptions,
            FormatPlaceholder, FormatSign, FormatTrait,
        },
    },
    lang_item::LangItemTarget,
    type_ref::{Mutability, Rawness},
};

impl<'db> ExprCollector<'db> {
    pub(super) fn collect_format_args(
        &mut self,
        f: ast::FormatArgsExpr,
        syntax_ptr: AstPtr<ast::Expr>,
    ) -> ExprId {
        let mut args = FormatArgumentsCollector::default();
        f.args().for_each(|arg| {
            args.add(FormatArgument {
                kind: match arg.arg_name() {
                    Some(name) => FormatArgumentKind::Named(Name::new_root(name.name().text())),
                    None => FormatArgumentKind::Normal,
                },
                expr: self.collect_expr_opt(arg.expr()),
            });
        });
        let template = f.template();
        let fmt_snippet = template.as_ref().and_then(|it| match it {
            ast::Expr::Literal(literal) => match literal.kind() {
                ast::LiteralKind::String(s) => Some(s.text().to_owned()),
                _ => None,
            },
            _ => None,
        });
        let mut mappings = vec![];
        let (fmt, hygiene) = match template.and_then(|template| {
            self.expand_macros_to_string(template.clone()).map(|it| (it, template))
        }) {
            Some(((s, is_direct_literal), template)) => {
                let call_ctx = SyntaxContext::root(self.def_map.edition());
                let hygiene = self.hygiene_id_for(s.syntax().text_range());
                let fmt = format_args::parse(
                    &s,
                    fmt_snippet,
                    args,
                    is_direct_literal,
                    |name, range| {
                        let expr_id = self.alloc_expr_desugared(Expr::Path(Path::from(name)));
                        if let Some(range) = range {
                            self.store
                                .template_map
                                .get_or_insert_with(Default::default)
                                .implicit_capture_to_source
                                .insert(
                                    expr_id,
                                    self.expander.in_file((AstPtr::new(&template), range)),
                                );
                        }
                        if !hygiene.is_root() {
                            self.store.ident_hygiene.insert(expr_id.into(), hygiene);
                        }
                        expr_id
                    },
                    |name, span| {
                        if let Some(span) = span {
                            mappings.push((span, name))
                        }
                    },
                    call_ctx,
                );
                (fmt, hygiene)
            }
            None => (
                FormatArgs {
                    template: Default::default(),
                    arguments: args.finish(),
                    orphans: Default::default(),
                },
                HygieneId::ROOT,
            ),
        };

        let idx = if self.lang_items().FormatCount.is_none() {
            self.collect_format_args_after_1_93_0_impl(syntax_ptr, fmt)
        } else {
            self.collect_format_args_before_1_93_0_impl(syntax_ptr, fmt)
        };

        self.store
            .template_map
            .get_or_insert_with(Default::default)
            .format_args_to_captures
            .insert(idx, (hygiene, mappings));
        idx
    }

    fn collect_format_args_after_1_93_0_impl(
        &mut self,
        syntax_ptr: AstPtr<ast::Expr>,
        fmt: FormatArgs,
    ) -> ExprId {
        let lang_items = self.lang_items();

        // Create a list of all _unique_ (argument, format trait) combinations.
        // E.g. "{0} {0:x} {0} {1}" -> [(0, Display), (0, LowerHex), (1, Display)]
        //
        // We use usize::MAX for arguments that don't exist, because that can never be a valid index
        // into the arguments array.
        let mut argmap = FxIndexSet::default();

        let mut incomplete_lit = String::new();

        let mut implicit_arg_index = 0;

        let mut bytecode = Vec::new();

        let template = if fmt.template.is_empty() {
            // Treat empty templates as a single literal piece (with an empty string),
            // so we produce `from_str("")` for those.
            &[FormatArgsPiece::Literal(sym::__empty)][..]
        } else {
            &fmt.template[..]
        };

        // See library/core/src/fmt/mod.rs for the format string encoding format.

        for (i, piece) in template.iter().enumerate() {
            match piece {
                FormatArgsPiece::Literal(sym) => {
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
                        let from_str = self.ty_rel_lang_path_desugared_expr(
                            lang_items.FormatArguments,
                            sym::from_str,
                        );
                        let sym =
                            if incomplete_lit.is_empty() { sym.clone() } else { Symbol::intern(s) };
                        let s = self.alloc_expr_desugared(Expr::Literal(Literal::String(sym)));
                        let from_str = self.alloc_expr(
                            Expr::Call { callee: from_str, args: Box::new([s]) },
                            syntax_ptr,
                        );
                        return if !fmt.arguments.arguments.is_empty() {
                            // With an incomplete format string (e.g. only an opening `{`), it's possible for `arguments`
                            // to be non-empty when reaching this code path.
                            self.alloc_expr(
                                Expr::Block {
                                    id: None,
                                    statements: fmt
                                        .arguments
                                        .arguments
                                        .iter()
                                        .map(|arg| Statement::Expr {
                                            expr: arg.expr,
                                            has_semi: true,
                                        })
                                        .collect(),
                                    tail: Some(from_str),
                                    label: None,
                                },
                                syntax_ptr,
                            )
                        } else {
                            from_str
                        };
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

                    let position = match &p.argument.index {
                        &Ok(it) => it,
                        Err(_) => usize::MAX,
                    };
                    let position = argmap
                        .insert_full((position, ArgumentType::Format(p.format_trait)))
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
                            let (indirect, val) = self.make_count_after_1_93_0(val, &mut argmap);
                            // Only encode if nonzero; zero is the default.
                            if indirect || val != 0 {
                                bytecode[i] |= 1 << 1 | (indirect as u8) << 4;
                                bytecode.extend_from_slice(&val.to_le_bytes());
                            }
                        }
                        if let Some(val) = &o.precision {
                            let (indirect, val) = self.make_count_after_1_93_0(val, &mut argmap);
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
            // FIXME: Emit an error.
            // ctx.dcx().span_err(macsp, "too many format arguments");
        }

        let arguments = &fmt.arguments.arguments[..];

        let (mut statements, args) = if arguments.is_empty() {
            // Generate:
            //     []
            (
                Vec::new(),
                self.alloc_expr_desugared(Expr::Array(Array::ElementList {
                    elements: Box::new([]),
                })),
            )
        } else {
            // Generate:
            //     super let args = (&arg0, &arg1, &…);
            let args_name = self.generate_new_name();
            let args_path = Path::from(args_name.clone());
            let args_binding = self.alloc_binding(
                args_name.clone(),
                BindingAnnotation::Unannotated,
                HygieneId::ROOT,
            );
            let args_pat = self.alloc_pat_desugared(Pat::Bind { id: args_binding, subpat: None });
            self.add_definition_to_binding(args_binding, args_pat);
            let elements = arguments
                .iter()
                .map(|arg| {
                    self.alloc_expr_desugared(Expr::Ref {
                        expr: arg.expr,
                        rawness: Rawness::Ref,
                        mutability: Mutability::Shared,
                    })
                })
                .collect();
            let args_tuple = self.alloc_expr_desugared(Expr::Tuple { exprs: elements });
            // FIXME: Make this a `super let` when we have this statement.
            let let_statement_1 = Statement::Let {
                pat: args_pat,
                type_ref: None,
                initializer: Some(args_tuple),
                else_branch: None,
            };

            // Generate:
            //     super let args = [
            //         <core::fmt::Argument>::new_display(args.0),
            //         <core::fmt::Argument>::new_lower_hex(args.1),
            //         <core::fmt::Argument>::new_debug(args.0),
            //         …
            //     ];
            let args = argmap
                .iter()
                .map(|&(arg_index, ty)| {
                    let args_ident_expr = self.alloc_expr_desugared(Expr::Path(args_path.clone()));
                    let arg = self.alloc_expr_desugared(Expr::Field {
                        expr: args_ident_expr,
                        name: Name::new_tuple_field(arg_index),
                    });
                    self.make_argument(arg, ty)
                })
                .collect();
            let args =
                self.alloc_expr_desugared(Expr::Array(Array::ElementList { elements: args }));
            let args_binding =
                self.alloc_binding(args_name, BindingAnnotation::Unannotated, HygieneId::ROOT);
            let args_pat = self.alloc_pat_desugared(Pat::Bind { id: args_binding, subpat: None });
            self.add_definition_to_binding(args_binding, args_pat);
            // FIXME: Make this a `super let` when we have this statement.
            let let_statement_2 = Statement::Let {
                pat: args_pat,
                type_ref: None,
                initializer: Some(args),
                else_branch: None,
            };
            (
                vec![let_statement_1, let_statement_2],
                self.alloc_expr_desugared(Expr::Path(args_path)),
            )
        };

        // Generate:
        //     unsafe {
        //         <core::fmt::Arguments>::new(b"…", &args)
        //     }
        let template = self
            .alloc_expr_desugared(Expr::Literal(Literal::ByteString(bytecode.into_boxed_slice())));
        let call = {
            let new = self.ty_rel_lang_path_desugared_expr(lang_items.FormatArguments, sym::new);
            let args = self.alloc_expr_desugared(Expr::Ref {
                expr: args,
                rawness: Rawness::Ref,
                mutability: Mutability::Shared,
            });
            self.alloc_expr_desugared(Expr::Call { callee: new, args: Box::new([template, args]) })
        };
        let call = self.alloc_expr(
            Expr::Unsafe { id: None, statements: Box::new([]), tail: Some(call) },
            syntax_ptr,
        );

        // We collect the unused expressions here so that we still infer them instead of
        // dropping them out of the expression tree. We cannot store them in the `Unsafe`
        // block because then unsafe blocks within them will get a false "unused unsafe"
        // diagnostic (rustc has a notion of builtin unsafe blocks, but we don't).
        statements
            .extend(fmt.orphans.into_iter().map(|expr| Statement::Expr { expr, has_semi: true }));

        if !statements.is_empty() {
            // Generate:
            //     {
            //         super let …
            //         super let …
            //         <core::fmt::Arguments>::new(…)
            //     }
            self.alloc_expr(
                Expr::Block {
                    id: None,
                    statements: statements.into_boxed_slice(),
                    tail: Some(call),
                    label: None,
                },
                syntax_ptr,
            )
        } else {
            call
        }
    }

    /// Get the value for a `width` or `precision` field.
    ///
    /// Returns the value and whether it is indirect (an indexed argument) or not.
    fn make_count_after_1_93_0(
        &self,
        count: &FormatCount,
        argmap: &mut FxIndexSet<(usize, ArgumentType)>,
    ) -> (bool, u16) {
        match count {
            FormatCount::Literal(n) => (false, *n),
            FormatCount::Argument(arg) => {
                let index = match &arg.index {
                    &Ok(it) => it,
                    Err(_) => usize::MAX,
                };
                (true, argmap.insert_full((index, ArgumentType::Usize)).0 as u16)
            }
        }
    }

    fn collect_format_args_before_1_93_0_impl(
        &mut self,
        syntax_ptr: AstPtr<ast::Expr>,
        fmt: FormatArgs,
    ) -> ExprId {
        // Create a list of all _unique_ (argument, format trait) combinations.
        // E.g. "{0} {0:x} {0} {1}" -> [(0, Display), (0, LowerHex), (1, Display)]
        let mut argmap = FxIndexSet::default();
        for piece in fmt.template.iter() {
            let FormatArgsPiece::Placeholder(placeholder) = piece else { continue };
            if let Ok(index) = placeholder.argument.index {
                argmap.insert((index, ArgumentType::Format(placeholder.format_trait)));
            }
        }

        let lit_pieces = fmt
            .template
            .iter()
            .enumerate()
            .filter_map(|(i, piece)| {
                match piece {
                    FormatArgsPiece::Literal(s) => {
                        Some(self.alloc_expr_desugared(Expr::Literal(Literal::String(s.clone()))))
                    }
                    &FormatArgsPiece::Placeholder(_) => {
                        // Inject empty string before placeholders when not already preceded by a literal piece.
                        if i == 0 || matches!(fmt.template[i - 1], FormatArgsPiece::Placeholder(_))
                        {
                            Some(self.alloc_expr_desugared(Expr::Literal(Literal::String(
                                Symbol::empty(),
                            ))))
                        } else {
                            None
                        }
                    }
                }
            })
            .collect();
        let lit_pieces =
            self.alloc_expr_desugared(Expr::Array(Array::ElementList { elements: lit_pieces }));
        let lit_pieces = self.alloc_expr_desugared(Expr::Ref {
            expr: lit_pieces,
            rawness: Rawness::Ref,
            mutability: Mutability::Shared,
        });
        let format_options = {
            // Generate:
            //     &[format_spec_0, format_spec_1, format_spec_2]
            let elements = fmt
                .template
                .iter()
                .filter_map(|piece| {
                    let FormatArgsPiece::Placeholder(placeholder) = piece else { return None };
                    Some(self.make_format_spec(placeholder, &mut argmap))
                })
                .collect();
            let array = self.alloc_expr_desugared(Expr::Array(Array::ElementList { elements }));
            self.alloc_expr_desugared(Expr::Ref {
                expr: array,
                rawness: Rawness::Ref,
                mutability: Mutability::Shared,
            })
        };

        // Assume that rustc version >= 1.89.0 iff lang item `format_arguments` exists
        // but `format_unsafe_arg` does not
        let lang_items = self.lang_items();
        let fmt_args = lang_items.FormatArguments;
        let fmt_unsafe_arg = lang_items.FormatUnsafeArg;
        let use_format_args_since_1_89_0 = fmt_args.is_some() && fmt_unsafe_arg.is_none();

        if use_format_args_since_1_89_0 {
            self.collect_format_args_after_1_89_0_impl(
                syntax_ptr,
                fmt,
                argmap,
                lit_pieces,
                format_options,
            )
        } else {
            self.collect_format_args_before_1_89_0_impl(
                syntax_ptr,
                fmt,
                argmap,
                lit_pieces,
                format_options,
            )
        }
    }

    /// `format_args!` expansion implementation for rustc versions < `1.89.0`
    fn collect_format_args_before_1_89_0_impl(
        &mut self,
        syntax_ptr: AstPtr<ast::Expr>,
        fmt: FormatArgs,
        argmap: FxIndexSet<(usize, ArgumentType)>,
        lit_pieces: ExprId,
        format_options: ExprId,
    ) -> ExprId {
        let arguments = &*fmt.arguments.arguments;

        let args = if arguments.is_empty() {
            let expr = self
                .alloc_expr_desugared(Expr::Array(Array::ElementList { elements: Box::default() }));
            self.alloc_expr_desugared(Expr::Ref {
                expr,
                rawness: Rawness::Ref,
                mutability: Mutability::Shared,
            })
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
            let args = argmap
                .iter()
                .map(|&(arg_index, ty)| {
                    let arg = self.alloc_expr_desugared(Expr::Ref {
                        expr: arguments[arg_index].expr,
                        rawness: Rawness::Ref,
                        mutability: Mutability::Shared,
                    });
                    self.make_argument(arg, ty)
                })
                .collect();
            let array =
                self.alloc_expr_desugared(Expr::Array(Array::ElementList { elements: args }));
            self.alloc_expr_desugared(Expr::Ref {
                expr: array,
                rawness: Rawness::Ref,
                mutability: Mutability::Shared,
            })
        };

        // Generate:
        //     <core::fmt::Arguments>::new_v1_formatted(
        //         lit_pieces,
        //         args,
        //         format_options,
        //         unsafe { ::core::fmt::UnsafeArg::new() }
        //     )

        let lang_items = self.lang_items();
        let new_v1_formatted =
            self.ty_rel_lang_path_desugared_expr(lang_items.FormatArguments, sym::new_v1_formatted);
        let unsafe_arg_new =
            self.ty_rel_lang_path_desugared_expr(lang_items.FormatUnsafeArg, sym::new);
        let unsafe_arg_new =
            self.alloc_expr_desugared(Expr::Call { callee: unsafe_arg_new, args: Box::default() });
        let mut unsafe_arg_new = self.alloc_expr_desugared(Expr::Unsafe {
            id: None,
            statements: Box::new([]),
            tail: Some(unsafe_arg_new),
        });
        if !fmt.orphans.is_empty() {
            unsafe_arg_new = self.alloc_expr_desugared(Expr::Block {
                id: None,
                // We collect the unused expressions here so that we still infer them instead of
                // dropping them out of the expression tree. We cannot store them in the `Unsafe`
                // block because then unsafe blocks within them will get a false "unused unsafe"
                // diagnostic (rustc has a notion of builtin unsafe blocks, but we don't).
                statements: fmt
                    .orphans
                    .into_iter()
                    .map(|expr| Statement::Expr { expr, has_semi: true })
                    .collect(),
                tail: Some(unsafe_arg_new),
                label: None,
            });
        }

        self.alloc_expr(
            Expr::Call {
                callee: new_v1_formatted,
                args: Box::new([lit_pieces, args, format_options, unsafe_arg_new]),
            },
            syntax_ptr,
        )
    }

    /// `format_args!` expansion implementation for rustc versions >= `1.89.0`,
    /// especially since [this PR](https://github.com/rust-lang/rust/pull/140748)
    fn collect_format_args_after_1_89_0_impl(
        &mut self,
        syntax_ptr: AstPtr<ast::Expr>,
        fmt: FormatArgs,
        argmap: FxIndexSet<(usize, ArgumentType)>,
        lit_pieces: ExprId,
        format_options: ExprId,
    ) -> ExprId {
        let arguments = &*fmt.arguments.arguments;

        let (let_stmts, args) = if arguments.is_empty() {
            (
                // Generate:
                //     []
                vec![],
                self.alloc_expr_desugared(Expr::Array(Array::ElementList {
                    elements: Box::default(),
                })),
            )
        } else if argmap.len() == 1 && arguments.len() == 1 {
            // Only one argument, so we don't need to make the `args` tuple.
            //
            // Generate:
            //     super let args = [<core::fmt::Arguments>::new_display(&arg)];
            let args = argmap
                .iter()
                .map(|&(arg_index, ty)| {
                    let ref_arg = self.alloc_expr_desugared(Expr::Ref {
                        expr: arguments[arg_index].expr,
                        rawness: Rawness::Ref,
                        mutability: Mutability::Shared,
                    });
                    self.make_argument(ref_arg, ty)
                })
                .collect();
            let args =
                self.alloc_expr_desugared(Expr::Array(Array::ElementList { elements: args }));
            let args_name = self.generate_new_name();
            let args_binding = self.alloc_binding(
                args_name.clone(),
                BindingAnnotation::Unannotated,
                HygieneId::ROOT,
            );
            let args_pat = self.alloc_pat_desugared(Pat::Bind { id: args_binding, subpat: None });
            self.add_definition_to_binding(args_binding, args_pat);
            // TODO: We don't have `super let` yet.
            let let_stmt = Statement::Let {
                pat: args_pat,
                type_ref: None,
                initializer: Some(args),
                else_branch: None,
            };
            (vec![let_stmt], self.alloc_expr_desugared(Expr::Path(args_name.into())))
        } else {
            // Generate:
            //     super let args = (&arg0, &arg1, &...);
            let args_name = self.generate_new_name();
            let args_binding = self.alloc_binding(
                args_name.clone(),
                BindingAnnotation::Unannotated,
                HygieneId::ROOT,
            );
            let args_pat = self.alloc_pat_desugared(Pat::Bind { id: args_binding, subpat: None });
            self.add_definition_to_binding(args_binding, args_pat);
            let elements = arguments
                .iter()
                .map(|arg| {
                    self.alloc_expr_desugared(Expr::Ref {
                        expr: arg.expr,
                        rawness: Rawness::Ref,
                        mutability: Mutability::Shared,
                    })
                })
                .collect();
            let args_tuple = self.alloc_expr_desugared(Expr::Tuple { exprs: elements });
            // TODO: We don't have `super let` yet
            let let_stmt1 = Statement::Let {
                pat: args_pat,
                type_ref: None,
                initializer: Some(args_tuple),
                else_branch: None,
            };

            // Generate:
            //     super let args = [
            //         <core::fmt::Argument>::new_display(args.0),
            //         <core::fmt::Argument>::new_lower_hex(args.1),
            //         <core::fmt::Argument>::new_debug(args.0),
            //         …
            //     ];
            let args = argmap
                .iter()
                .map(|&(arg_index, ty)| {
                    let args_ident_expr =
                        self.alloc_expr_desugared(Expr::Path(args_name.clone().into()));
                    let arg = self.alloc_expr_desugared(Expr::Field {
                        expr: args_ident_expr,
                        name: Name::new_tuple_field(arg_index),
                    });
                    self.make_argument(arg, ty)
                })
                .collect();
            let array =
                self.alloc_expr_desugared(Expr::Array(Array::ElementList { elements: args }));
            let args_binding = self.alloc_binding(
                args_name.clone(),
                BindingAnnotation::Unannotated,
                HygieneId::ROOT,
            );
            let args_pat = self.alloc_pat_desugared(Pat::Bind { id: args_binding, subpat: None });
            self.add_definition_to_binding(args_binding, args_pat);
            let let_stmt2 = Statement::Let {
                pat: args_pat,
                type_ref: None,
                initializer: Some(array),
                else_branch: None,
            };
            (vec![let_stmt1, let_stmt2], self.alloc_expr_desugared(Expr::Path(args_name.into())))
        };

        // Generate:
        //     &args
        let args = self.alloc_expr_desugared(Expr::Ref {
            expr: args,
            rawness: Rawness::Ref,
            mutability: Mutability::Shared,
        });

        let call_block = {
            // Generate:
            //     unsafe {
            //         <core::fmt::Arguments>::new_v1_formatted(
            //             lit_pieces,
            //             args,
            //             format_options,
            //         )
            //     }

            let new_v1_formatted = self.ty_rel_lang_path_desugared_expr(
                self.lang_items().FormatArguments,
                sym::new_v1_formatted,
            );
            let args = [lit_pieces, args, format_options];
            let call = self
                .alloc_expr_desugared(Expr::Call { callee: new_v1_formatted, args: args.into() });

            Expr::Unsafe { id: None, statements: Box::default(), tail: Some(call) }
        };

        if !let_stmts.is_empty() {
            // Generate:
            //     {
            //         super let …
            //         super let …
            //         <core::fmt::Arguments>::new_…(…)
            //     }
            let call = self.alloc_expr_desugared(call_block);
            self.alloc_expr(
                Expr::Block {
                    id: None,
                    statements: let_stmts.into(),
                    tail: Some(call),
                    label: None,
                },
                syntax_ptr,
            )
        } else {
            self.alloc_expr(call_block, syntax_ptr)
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
    fn make_format_spec(
        &mut self,
        placeholder: &FormatPlaceholder,
        argmap: &mut FxIndexSet<(usize, ArgumentType)>,
    ) -> ExprId {
        let lang_items = self.lang_items();
        let position = match placeholder.argument.index {
            Ok(arg_index) => {
                let (i, _) =
                    argmap.insert_full((arg_index, ArgumentType::Format(placeholder.format_trait)));
                self.alloc_expr_desugared(Expr::Literal(Literal::Uint(
                    i as u128,
                    Some(BuiltinUint::Usize),
                )))
            }
            Err(_) => self.missing_expr(),
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

        let precision_expr = self.make_count_before_1_93_0(precision, argmap);
        let width_expr = self.make_count_before_1_93_0(width, argmap);

        if self.krate.workspace_data(self.db).is_atleast_187() {
            // These need to match the constants in library/core/src/fmt/rt.rs.
            let align = match alignment {
                Some(FormatAlignment::Left) => 0,
                Some(FormatAlignment::Right) => 1,
                Some(FormatAlignment::Center) => 2,
                None => 3,
            };
            // This needs to match `Flag` in library/core/src/fmt/rt.rs.
            let flags = fill.unwrap_or(' ') as u32
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
            let flags = self.alloc_expr_desugared(Expr::Literal(Literal::Uint(
                flags as u128,
                Some(BuiltinUint::U32),
            )));

            let position =
                RecordLitField { name: Name::new_symbol_root(sym::position), expr: position };
            let flags = RecordLitField { name: Name::new_symbol_root(sym::flags), expr: flags };
            let precision = RecordLitField {
                name: Name::new_symbol_root(sym::precision),
                expr: precision_expr,
            };
            let width =
                RecordLitField { name: Name::new_symbol_root(sym::width), expr: width_expr };
            self.alloc_expr_desugared(Expr::RecordLit {
                path: self.lang_path(lang_items.FormatPlaceholder).map(Box::new),
                fields: Box::new([position, flags, precision, width]),
                spread: RecordSpread::None,
            })
        } else {
            let format_placeholder_new =
                self.ty_rel_lang_path_desugared_expr(lang_items.FormatPlaceholder, sym::new);
            // This needs to match `Flag` in library/core/src/fmt/rt.rs.
            let flags: u32 = ((sign == Some(FormatSign::Plus)) as u32)
                | (((sign == Some(FormatSign::Minus)) as u32) << 1)
                | ((alternate as u32) << 2)
                | ((zero_pad as u32) << 3)
                | (((debug_hex == Some(FormatDebugHex::Lower)) as u32) << 4)
                | (((debug_hex == Some(FormatDebugHex::Upper)) as u32) << 5);
            let flags = self.alloc_expr_desugared(Expr::Literal(Literal::Uint(
                flags as u128,
                Some(BuiltinUint::U32),
            )));
            let fill = self.alloc_expr_desugared(Expr::Literal(Literal::Char(fill.unwrap_or(' '))));
            let align = self.ty_rel_lang_path_desugared_expr(
                lang_items.FormatAlignment,
                match alignment {
                    Some(FormatAlignment::Left) => sym::Left,
                    Some(FormatAlignment::Right) => sym::Right,
                    Some(FormatAlignment::Center) => sym::Center,
                    None => sym::Unknown,
                },
            );
            self.alloc_expr_desugared(Expr::Call {
                callee: format_placeholder_new,
                args: Box::new([position, fill, align, flags, precision_expr, width_expr]),
            })
        }
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
    fn make_count_before_1_93_0(
        &mut self,
        count: &Option<FormatCount>,
        argmap: &mut FxIndexSet<(usize, ArgumentType)>,
    ) -> ExprId {
        let lang_items = self.lang_items();
        match count {
            Some(FormatCount::Literal(n)) => {
                let args = self.alloc_expr_desugared(Expr::Literal(Literal::Uint(
                    *n as u128,
                    // FIXME: Change this to Some(BuiltinUint::U16) once we drop support for toolchains < 1.88
                    None,
                )));
                let count_is =
                    self.ty_rel_lang_path_desugared_expr(lang_items.FormatCount, sym::Is);
                self.alloc_expr_desugared(Expr::Call { callee: count_is, args: Box::new([args]) })
            }
            Some(FormatCount::Argument(arg)) => {
                if let Ok(arg_index) = arg.index {
                    let (i, _) = argmap.insert_full((arg_index, ArgumentType::Usize));

                    let args = self.alloc_expr_desugared(Expr::Literal(Literal::Uint(
                        i as u128,
                        Some(BuiltinUint::Usize),
                    )));
                    let count_param =
                        self.ty_rel_lang_path_desugared_expr(lang_items.FormatCount, sym::Param);
                    self.alloc_expr_desugared(Expr::Call {
                        callee: count_param,
                        args: Box::new([args]),
                    })
                } else {
                    // FIXME: This drops arg causing it to potentially not be resolved/type checked
                    // when typing?
                    self.missing_expr()
                }
            }
            None => match self.ty_rel_lang_path(lang_items.FormatCount, sym::Implied) {
                Some(count_param) => self.alloc_expr_desugared(Expr::Path(count_param)),
                None => self.missing_expr(),
            },
        }
    }

    /// Generate a hir expression representing an argument to a format_args invocation.
    ///
    /// Generates:
    ///
    /// ```text
    ///     <core::fmt::Argument>::new_…(arg)
    /// ```
    fn make_argument(&mut self, arg: ExprId, ty: ArgumentType) -> ExprId {
        use ArgumentType::*;
        use FormatTrait::*;

        let new_fn = self.ty_rel_lang_path_desugared_expr(
            self.lang_items().FormatArgument,
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
        );
        self.alloc_expr_desugared(Expr::Call { callee: new_fn, args: Box::new([arg]) })
    }

    fn ty_rel_lang_path_desugared_expr(
        &mut self,
        lang: Option<impl Into<LangItemTarget>>,
        relative_name: Symbol,
    ) -> ExprId {
        self.alloc_expr_desugared(self.ty_rel_lang_path_expr(lang, relative_name))
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
enum ArgumentType {
    Format(FormatTrait),
    Usize,
}
