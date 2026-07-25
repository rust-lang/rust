//! Lowering of `format_args!()`.

use base_db::FxIndexSet;
use hir_expand::name::Name;
use intern::{Symbol, sym};
use span::SyntaxContext;
use syntax::{AstPtr, AstToken as _, ast};

use crate::{
    expr_store::{HygieneId, lower::ExprCollector, path::Path},
    hir::{
        Array, BindingAnnotation, Expr, ExprId, Literal, Pat, Statement,
        format_args::{
            self, FormatAlignment, FormatArgs, FormatArgsPiece, FormatArgument, FormatArgumentKind,
            FormatArgumentsCollector, FormatCount, FormatDebugHex, FormatSign, FormatTrait,
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
            let expr = arg.expr();
            args.add(FormatArgument {
                kind: match arg.arg_name() {
                    Some(name) => FormatArgumentKind::Named(Name::new_root(name.name().text())),
                    None => FormatArgumentKind::Normal,
                },
                syntax: expr.as_ref().map(AstPtr::new),
                expr: self.collect_expr_opt(expr),
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
                let template_ptr = AstPtr::new(&template);
                let fmt = format_args::parse(
                    &s,
                    template_ptr,
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
                                .insert(expr_id, self.expander.in_file((template_ptr, range)));
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

        let idx = self.collect_format_args_impl(syntax_ptr, fmt);

        self.store
            .template_map
            .get_or_insert_with(Default::default)
            .format_args_to_captures
            .insert(idx, (hygiene, mappings));
        idx
    }

    // This is in separate functions because historically, changes in format_args lowering have forced us to change this
    // function but not its caller, and for some time support both versions.
    fn collect_format_args_impl(
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
                    let arg_ptr = arguments.get(arg_index).and_then(|it| it.syntax);
                    self.make_argument(arg_ptr, arg, ty)
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

    /// Generate a hir expression representing an argument to a format_args invocation.
    ///
    /// Generates:
    ///
    /// ```text
    ///     <core::fmt::Argument>::new_…(arg)
    /// ```
    fn make_argument(
        &mut self,
        arg_ptr: Option<AstPtr<ast::Expr>>,
        arg: ExprId,
        ty: ArgumentType,
    ) -> ExprId {
        use ArgumentType::*;
        use FormatTrait::*;

        let new_fn = self.ty_rel_lang_path(
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
        let new_fn = match new_fn {
            Some(new_fn) => {
                let new_fn = self.store.exprs.alloc(Expr::Path(new_fn));
                if let Some(arg_ptr) = arg_ptr {
                    // Trait errors (the argument does not implement the expected fmt trait) will show
                    // on this path, so to not end up with synthetic syntax we insert this mapping. We
                    // don't want to insert the other way's mapping in order to not override the source
                    // for the argument.
                    self.store
                        .expr_map_back
                        .insert(new_fn, self.expander.in_file(arg_ptr.wrap_left()));
                }
                new_fn
            }
            None => self.missing_expr(),
        };
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
