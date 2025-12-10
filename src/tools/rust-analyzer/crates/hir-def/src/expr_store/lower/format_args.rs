//! Lowering of `format_args!()`.

use base_db::FxIndexSet;
use hir_expand::name::{AsName, Name};
use intern::{Symbol, sym};
use syntax::{
    AstPtr, AstToken as _,
    ast::{self, HasName},
};

use crate::{
    builtin_type::BuiltinUint,
    expr_store::{HygieneId, lower::ExprCollector, path::Path},
    hir::{
        Array, BindingAnnotation, Expr, ExprId, Literal, Pat, RecordLitField, Statement,
        format_args::{
            self, FormatAlignment, FormatArgs, FormatArgsPiece, FormatArgument, FormatArgumentKind,
            FormatArgumentsCollector, FormatCount, FormatDebugHex, FormatOptions,
            FormatPlaceholder, FormatSign, FormatTrait,
        },
    },
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
                kind: match arg.name() {
                    Some(name) => FormatArgumentKind::Named(name.as_name()),
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
                let call_ctx = self.expander.call_syntax_ctx();
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

        let idx = if use_format_args_since_1_89_0 {
            self.collect_format_args_impl(syntax_ptr, fmt, argmap, lit_pieces, format_options)
        } else {
            self.collect_format_args_before_1_89_0_impl(
                syntax_ptr,
                fmt,
                argmap,
                lit_pieces,
                format_options,
            )
        };

        self.store
            .template_map
            .get_or_insert_with(Default::default)
            .format_args_to_captures
            .insert(idx, (hygiene, mappings));
        idx
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
        let new_v1_formatted = self.ty_rel_lang_path(
            lang_items.FormatArguments,
            Name::new_symbol_root(sym::new_v1_formatted),
        );
        let unsafe_arg_new =
            self.ty_rel_lang_path(lang_items.FormatUnsafeArg, Name::new_symbol_root(sym::new));
        let new_v1_formatted =
            self.alloc_expr_desugared(new_v1_formatted.map_or(Expr::Missing, Expr::Path));

        let unsafe_arg_new =
            self.alloc_expr_desugared(unsafe_arg_new.map_or(Expr::Missing, Expr::Path));
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
    fn collect_format_args_impl(
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
            let args_name = Name::new_symbol_root(sym::args);
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
            let args_name = Name::new_symbol_root(sym::args);
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

            let new_v1_formatted = self.ty_rel_lang_path(
                self.lang_items().FormatArguments,
                Name::new_symbol_root(sym::new_v1_formatted),
            );
            let new_v1_formatted =
                self.alloc_expr_desugared(new_v1_formatted.map_or(Expr::Missing, Expr::Path));
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

        let precision_expr = self.make_count(precision, argmap);
        let width_expr = self.make_count(width, argmap);

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
                spread: None,
            })
        } else {
            let format_placeholder_new = {
                let format_placeholder_new = self.ty_rel_lang_path(
                    lang_items.FormatPlaceholder,
                    Name::new_symbol_root(sym::new),
                );
                match format_placeholder_new {
                    Some(path) => self.alloc_expr_desugared(Expr::Path(path)),
                    None => self.missing_expr(),
                }
            };
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
            let align = {
                let align = self.ty_rel_lang_path(
                    lang_items.FormatAlignment,
                    match alignment {
                        Some(FormatAlignment::Left) => Name::new_symbol_root(sym::Left),
                        Some(FormatAlignment::Right) => Name::new_symbol_root(sym::Right),
                        Some(FormatAlignment::Center) => Name::new_symbol_root(sym::Center),
                        None => Name::new_symbol_root(sym::Unknown),
                    },
                );
                match align {
                    Some(path) => self.alloc_expr_desugared(Expr::Path(path)),
                    None => self.missing_expr(),
                }
            };
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
    fn make_count(
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
                let count_is = match self
                    .ty_rel_lang_path(lang_items.FormatCount, Name::new_symbol_root(sym::Is))
                {
                    Some(count_is) => self.alloc_expr_desugared(Expr::Path(count_is)),
                    None => self.missing_expr(),
                };
                self.alloc_expr_desugared(Expr::Call { callee: count_is, args: Box::new([args]) })
            }
            Some(FormatCount::Argument(arg)) => {
                if let Ok(arg_index) = arg.index {
                    let (i, _) = argmap.insert_full((arg_index, ArgumentType::Usize));

                    let args = self.alloc_expr_desugared(Expr::Literal(Literal::Uint(
                        i as u128,
                        Some(BuiltinUint::Usize),
                    )));
                    let count_param = match self
                        .ty_rel_lang_path(lang_items.FormatCount, Name::new_symbol_root(sym::Param))
                    {
                        Some(count_param) => self.alloc_expr_desugared(Expr::Path(count_param)),
                        None => self.missing_expr(),
                    };
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
            None => match self
                .ty_rel_lang_path(lang_items.FormatCount, Name::new_symbol_root(sym::Implied))
            {
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

        let new_fn = match self.ty_rel_lang_path(
            self.lang_items().FormatArgument,
            Name::new_symbol_root(match ty {
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
            }),
        ) {
            Some(new_fn) => self.alloc_expr_desugared(Expr::Path(new_fn)),
            None => self.missing_expr(),
        };
        self.alloc_expr_desugared(Expr::Call { callee: new_fn, args: Box::new([arg]) })
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
enum ArgumentType {
    Format(FormatTrait),
    Usize,
}
