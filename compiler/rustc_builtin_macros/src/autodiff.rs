//! This module contains the implementation of the `#[autodiff]` attribute.
//! Currently our linter isn't smart enough to see that each import is used in one of the two
//! configs (autodiff enabled or disabled), so we have to add cfg's to each import.
//! FIXME(ZuseZ4): Remove this once we have a smarter linter.

mod llvm_enzyme {
    use std::str::FromStr;
    use std::string::String;

    use rustc_ast::expand::autodiff_attrs::{
        AutoDiffAttrs, DiffActivity, DiffMode, valid_input_activity, valid_ret_activity,
        valid_ty_for_activity,
    };
    use rustc_ast::ptr::P;
    use rustc_ast::token::{Lit, LitKind, Token, TokenKind};
    use rustc_ast::tokenstream::*;
    use rustc_ast::visit::AssocCtxt::*;
    use rustc_ast::{
        self as ast, AssocItemKind, BindingMode, ExprKind, FnRetTy, FnSig, Generics, ItemKind,
        MetaItemInner, PatKind, QSelf, TyKind, Visibility,
    };
    use rustc_expand::base::{Annotatable, ExtCtxt};
    use rustc_span::{Ident, Span, Symbol, kw, sym};
    use thin_vec::{ThinVec, thin_vec};
    use tracing::{debug, trace};

    use crate::errors;

    pub(crate) fn outer_normal_attr(
        kind: &P<rustc_ast::NormalAttr>,
        id: rustc_ast::AttrId,
        span: Span,
    ) -> rustc_ast::Attribute {
        let style = rustc_ast::AttrStyle::Outer;
        let kind = rustc_ast::AttrKind::Normal(kind.clone());
        rustc_ast::Attribute { kind, id, style, span }
    }

    // If we have a default `()` return type or explicitley `()` return type,
    // then we often can skip doing some work.
    fn has_ret(ty: &FnRetTy) -> bool {
        match ty {
            FnRetTy::Ty(ty) => !ty.kind.is_unit(),
            FnRetTy::Default(_) => false,
        }
    }
    fn first_ident(x: &MetaItemInner) -> rustc_span::Ident {
        if let Some(l) = x.lit() {
            match l.kind {
                ast::LitKind::Int(val, _) => {
                    // get an Ident from a lit
                    return rustc_span::Ident::from_str(val.get().to_string().as_str());
                }
                _ => {}
            }
        }

        let segments = &x.meta_item().unwrap().path.segments;
        assert!(segments.len() == 1);
        segments[0].ident
    }

    fn name(x: &MetaItemInner) -> String {
        first_ident(x).name.to_string()
    }

    fn width(x: &MetaItemInner) -> Option<u128> {
        let lit = x.lit()?;
        match lit.kind {
            ast::LitKind::Int(x, _) => Some(x.get()),
            _ => return None,
        }
    }

    // Get information about the function the macro is applied to
    fn extract_item_info(iitem: &P<ast::Item>) -> Option<(Visibility, FnSig, Ident, Generics)> {
        match &iitem.kind {
            ItemKind::Fn(box ast::Fn { sig, ident, generics, .. }) => {
                Some((iitem.vis.clone(), sig.clone(), ident.clone(), generics.clone()))
            }
            _ => None,
        }
    }

    pub(crate) fn from_ast(
        ecx: &mut ExtCtxt<'_>,
        meta_item: &ThinVec<MetaItemInner>,
        has_ret: bool,
        mode: DiffMode,
    ) -> AutoDiffAttrs {
        let dcx = ecx.sess.dcx();

        // Now we check, whether the user wants autodiff in batch/vector mode, or scalar mode.
        // If he doesn't specify an integer (=width), we default to scalar mode, thus width=1.
        let mut first_activity = 1;

        let width = if let [_, x, ..] = &meta_item[..]
            && let Some(x) = width(x)
        {
            first_activity = 2;
            match x.try_into() {
                Ok(x) => x,
                Err(_) => {
                    dcx.emit_err(errors::AutoDiffInvalidWidth {
                        span: meta_item[1].span(),
                        width: x,
                    });
                    return AutoDiffAttrs::error();
                }
            }
        } else {
            1
        };

        let mut activities: Vec<DiffActivity> = vec![];
        let mut errors = false;
        for x in &meta_item[first_activity..] {
            let activity_str = name(&x);
            let res = DiffActivity::from_str(&activity_str);
            match res {
                Ok(x) => activities.push(x),
                Err(_) => {
                    dcx.emit_err(errors::AutoDiffUnknownActivity {
                        span: x.span(),
                        act: activity_str,
                    });
                    errors = true;
                }
            };
        }
        if errors {
            return AutoDiffAttrs::error();
        }

        // If a return type exist, we need to split the last activity,
        // otherwise we return None as placeholder.
        let (ret_activity, input_activity) = if has_ret {
            let Some((last, rest)) = activities.split_last() else {
                unreachable!(
                    "should not be reachable because we counted the number of activities previously"
                );
            };
            (last, rest)
        } else {
            (&DiffActivity::None, activities.as_slice())
        };

        AutoDiffAttrs {
            mode,
            width,
            ret_activity: *ret_activity,
            input_activity: input_activity.to_vec(),
        }
    }

    fn meta_item_inner_to_ts(t: &MetaItemInner, ts: &mut Vec<TokenTree>) {
        let comma: Token = Token::new(TokenKind::Comma, Span::default());
        let val = first_ident(t);
        let t = Token::from_ast_ident(val);
        ts.push(TokenTree::Token(t, Spacing::Joint));
        ts.push(TokenTree::Token(comma.clone(), Spacing::Alone));
    }

    pub(crate) fn expand_forward(
        ecx: &mut ExtCtxt<'_>,
        expand_span: Span,
        meta_item: &ast::MetaItem,
        item: Annotatable,
    ) -> Vec<Annotatable> {
        expand_with_mode(ecx, expand_span, meta_item, item, DiffMode::Forward)
    }

    pub(crate) fn expand_reverse(
        ecx: &mut ExtCtxt<'_>,
        expand_span: Span,
        meta_item: &ast::MetaItem,
        item: Annotatable,
    ) -> Vec<Annotatable> {
        expand_with_mode(ecx, expand_span, meta_item, item, DiffMode::Reverse)
    }

    /// We expand the autodiff macro to generate a new placeholder function which passes
    /// type-checking and can be called by users. The function body of the placeholder function will
    /// later be replaced on LLVM-IR level, so the design of the body is less important and for now
    /// should just prevent early inlining and optimizations which alter the function signature.
    /// The exact signature of the generated function depends on the configuration provided by the
    /// user, but here is an example:
    ///
    /// ```
    /// #[autodiff(cos_box, Reverse, Duplicated, Active)]
    /// fn sin(x: &Box<f32>) -> f32 {
    ///     f32::sin(**x)
    /// }
    /// ```
    /// which becomes expanded to:
    /// ```
    /// #[rustc_autodiff]
    /// #[inline(never)]
    /// fn sin(x: &Box<f32>) -> f32 {
    ///     f32::sin(**x)
    /// }
    /// #[rustc_autodiff(Reverse, Duplicated, Active)]
    /// #[inline(never)]
    /// fn cos_box(x: &Box<f32>, dx: &mut Box<f32>, dret: f32) -> f32 {
    ///     unsafe {
    ///         asm!("NOP");
    ///     };
    ///     ::core::hint::black_box(sin(x));
    ///     ::core::hint::black_box((dx, dret));
    ///     ::core::hint::black_box(sin(x))
    /// }
    /// ```
    /// FIXME(ZuseZ4): Once autodiff is enabled by default, make this a doc comment which is checked
    /// in CI.
    pub(crate) fn expand_with_mode(
        ecx: &mut ExtCtxt<'_>,
        expand_span: Span,
        meta_item: &ast::MetaItem,
        mut item: Annotatable,
        mode: DiffMode,
    ) -> Vec<Annotatable> {
        if cfg!(not(llvm_enzyme)) {
            ecx.sess.dcx().emit_err(errors::AutoDiffSupportNotBuild { span: meta_item.span });
            return vec![item];
        }
        let dcx = ecx.sess.dcx();

        // first get information about the annotable item: visibility, signature, name and generic
        // parameters.
        // these will be used to generate the differentiated version of the function
        let Some((vis, sig, primal, generics)) = (match &item {
            Annotatable::Item(iitem) => extract_item_info(iitem),
            Annotatable::Stmt(stmt) => match &stmt.kind {
                ast::StmtKind::Item(iitem) => extract_item_info(iitem),
                _ => None,
            },
            Annotatable::AssocItem(assoc_item, Impl { .. }) => match &assoc_item.kind {
                ast::AssocItemKind::Fn(box ast::Fn { sig, ident, generics, .. }) => {
                    Some((assoc_item.vis.clone(), sig.clone(), ident.clone(), generics.clone()))
                }
                _ => None,
            },
            _ => None,
        }) else {
            dcx.emit_err(errors::AutoDiffInvalidApplication { span: item.span() });
            return vec![item];
        };

        let meta_item_vec: ThinVec<MetaItemInner> = match meta_item.kind {
            ast::MetaItemKind::List(ref vec) => vec.clone(),
            _ => {
                dcx.emit_err(errors::AutoDiffMissingConfig { span: item.span() });
                return vec![item];
            }
        };

        let has_ret = has_ret(&sig.decl.output);
        let sig_span = ecx.with_call_site_ctxt(sig.span);

        // create TokenStream from vec elemtents:
        // meta_item doesn't have a .tokens field
        let mut ts: Vec<TokenTree> = vec![];
        if meta_item_vec.len() < 1 {
            // At the bare minimum, we need a fnc name.
            dcx.emit_err(errors::AutoDiffMissingConfig { span: item.span() });
            return vec![item];
        }

        let mode_symbol = match mode {
            DiffMode::Forward => sym::Forward,
            DiffMode::Reverse => sym::Reverse,
            _ => unreachable!("Unsupported mode: {:?}", mode),
        };

        // Insert mode token
        let mode_token = Token::new(TokenKind::Ident(mode_symbol, false.into()), Span::default());
        ts.insert(0, TokenTree::Token(mode_token, Spacing::Joint));
        ts.insert(
            1,
            TokenTree::Token(Token::new(TokenKind::Comma, Span::default()), Spacing::Alone),
        );

        // Now, if the user gave a width (vector aka batch-mode ad), then we copy it.
        // If it is not given, we default to 1 (scalar mode).
        let start_position;
        let kind: LitKind = LitKind::Integer;
        let symbol;
        if meta_item_vec.len() >= 2
            && let Some(width) = width(&meta_item_vec[1])
        {
            start_position = 2;
            symbol = Symbol::intern(&width.to_string());
        } else {
            start_position = 1;
            symbol = sym::integer(1);
        }

        let l: Lit = Lit { kind, symbol, suffix: None };
        let t = Token::new(TokenKind::Literal(l), Span::default());
        let comma = Token::new(TokenKind::Comma, Span::default());
        ts.push(TokenTree::Token(t, Spacing::Joint));
        ts.push(TokenTree::Token(comma.clone(), Spacing::Alone));

        for t in meta_item_vec.clone()[start_position..].iter() {
            meta_item_inner_to_ts(t, &mut ts);
        }

        if !has_ret {
            // We don't want users to provide a return activity if the function doesn't return anything.
            // For simplicity, we just add a dummy token to the end of the list.
            let t = Token::new(TokenKind::Ident(sym::None, false.into()), Span::default());
            ts.push(TokenTree::Token(t, Spacing::Joint));
            ts.push(TokenTree::Token(comma, Spacing::Alone));
        }
        // We remove the last, trailing comma.
        ts.pop();
        let ts: TokenStream = TokenStream::from_iter(ts);

        let x: AutoDiffAttrs = from_ast(ecx, &meta_item_vec, has_ret, mode);
        if !x.is_active() {
            // We encountered an error, so we return the original item.
            // This allows us to potentially parse other attributes.
            return vec![item];
        }
        let span = ecx.with_def_site_ctxt(expand_span);

        let n_active: u32 = x
            .input_activity
            .iter()
            .filter(|a| **a == DiffActivity::Active || **a == DiffActivity::ActiveOnly)
            .count() as u32;
        let (d_sig, new_args, idents, errored) = gen_enzyme_decl(ecx, &sig, &x, span);
        let d_body = gen_enzyme_body(
            ecx, &x, n_active, &sig, &d_sig, primal, &new_args, span, sig_span, idents, errored,
            &generics,
        );

        // The first element of it is the name of the function to be generated
        let asdf = Box::new(ast::Fn {
            defaultness: ast::Defaultness::Final,
            sig: d_sig,
            ident: first_ident(&meta_item_vec[0]),
            generics,
            contract: None,
            body: Some(d_body),
            define_opaque: None,
        });
        let mut rustc_ad_attr =
            P(ast::NormalAttr::from_ident(Ident::with_dummy_span(sym::rustc_autodiff)));

        let ts2: Vec<TokenTree> = vec![TokenTree::Token(
            Token::new(TokenKind::Ident(sym::never, false.into()), span),
            Spacing::Joint,
        )];
        let never_arg = ast::DelimArgs {
            dspan: DelimSpan::from_single(span),
            delim: ast::token::Delimiter::Parenthesis,
            tokens: TokenStream::from_iter(ts2),
        };
        let inline_item = ast::AttrItem {
            unsafety: ast::Safety::Default,
            path: ast::Path::from_ident(Ident::with_dummy_span(sym::inline)),
            args: ast::AttrArgs::Delimited(never_arg),
            tokens: None,
        };
        let inline_never_attr = P(ast::NormalAttr { item: inline_item, tokens: None });
        let new_id = ecx.sess.psess.attr_id_generator.mk_attr_id();
        let attr = outer_normal_attr(&rustc_ad_attr, new_id, span);
        let new_id = ecx.sess.psess.attr_id_generator.mk_attr_id();
        let inline_never = outer_normal_attr(&inline_never_attr, new_id, span);

        // We're avoid duplicating the attributes `#[rustc_autodiff]` and `#[inline(never)]`.
        fn same_attribute(attr: &ast::AttrKind, item: &ast::AttrKind) -> bool {
            match (attr, item) {
                (ast::AttrKind::Normal(a), ast::AttrKind::Normal(b)) => {
                    let a = &a.item.path;
                    let b = &b.item.path;
                    a.segments.len() == b.segments.len()
                        && a.segments.iter().zip(b.segments.iter()).all(|(a, b)| a.ident == b.ident)
                }
                _ => false,
            }
        }

        // Don't add it multiple times:
        let orig_annotatable: Annotatable = match item {
            Annotatable::Item(ref mut iitem) => {
                if !iitem.attrs.iter().any(|a| same_attribute(&a.kind, &attr.kind)) {
                    iitem.attrs.push(attr);
                }
                if !iitem.attrs.iter().any(|a| same_attribute(&a.kind, &inline_never.kind)) {
                    iitem.attrs.push(inline_never.clone());
                }
                Annotatable::Item(iitem.clone())
            }
            Annotatable::AssocItem(ref mut assoc_item, i @ Impl { .. }) => {
                if !assoc_item.attrs.iter().any(|a| same_attribute(&a.kind, &attr.kind)) {
                    assoc_item.attrs.push(attr);
                }
                if !assoc_item.attrs.iter().any(|a| same_attribute(&a.kind, &inline_never.kind)) {
                    assoc_item.attrs.push(inline_never.clone());
                }
                Annotatable::AssocItem(assoc_item.clone(), i)
            }
            Annotatable::Stmt(ref mut stmt) => {
                match stmt.kind {
                    ast::StmtKind::Item(ref mut iitem) => {
                        if !iitem.attrs.iter().any(|a| same_attribute(&a.kind, &attr.kind)) {
                            iitem.attrs.push(attr);
                        }
                        if !iitem.attrs.iter().any(|a| same_attribute(&a.kind, &inline_never.kind))
                        {
                            iitem.attrs.push(inline_never.clone());
                        }
                    }
                    _ => unreachable!("stmt kind checked previously"),
                };

                Annotatable::Stmt(stmt.clone())
            }
            _ => {
                unreachable!("annotatable kind checked previously")
            }
        };
        // Now update for d_fn
        rustc_ad_attr.item.args = rustc_ast::AttrArgs::Delimited(rustc_ast::DelimArgs {
            dspan: DelimSpan::dummy(),
            delim: rustc_ast::token::Delimiter::Parenthesis,
            tokens: ts,
        });

        let d_attr = outer_normal_attr(&rustc_ad_attr, new_id, span);
        let d_annotatable = match &item {
            Annotatable::AssocItem(_, _) => {
                let assoc_item: AssocItemKind = ast::AssocItemKind::Fn(asdf);
                let d_fn = P(ast::AssocItem {
                    attrs: thin_vec![d_attr, inline_never],
                    id: ast::DUMMY_NODE_ID,
                    span,
                    vis,
                    kind: assoc_item,
                    tokens: None,
                });
                Annotatable::AssocItem(d_fn, Impl { of_trait: false })
            }
            Annotatable::Item(_) => {
                let mut d_fn = ecx.item(span, thin_vec![d_attr, inline_never], ItemKind::Fn(asdf));
                d_fn.vis = vis;

                Annotatable::Item(d_fn)
            }
            Annotatable::Stmt(_) => {
                let mut d_fn = ecx.item(span, thin_vec![d_attr, inline_never], ItemKind::Fn(asdf));
                d_fn.vis = vis;

                Annotatable::Stmt(P(ast::Stmt {
                    id: ast::DUMMY_NODE_ID,
                    kind: ast::StmtKind::Item(d_fn),
                    span,
                }))
            }
            _ => {
                unreachable!("item kind checked previously")
            }
        };

        return vec![orig_annotatable, d_annotatable];
    }

    // shadow arguments (the extra ones which were not in the original (primal) function), in reverse mode must be
    // mutable references or ptrs, because Enzyme will write into them.
    fn assure_mut_ref(ty: &ast::Ty) -> ast::Ty {
        let mut ty = ty.clone();
        match ty.kind {
            TyKind::Ptr(ref mut mut_ty) => {
                mut_ty.mutbl = ast::Mutability::Mut;
            }
            TyKind::Ref(_, ref mut mut_ty) => {
                mut_ty.mutbl = ast::Mutability::Mut;
            }
            _ => {
                panic!("unsupported type: {:?}", ty);
            }
        }
        ty
    }

    // Will generate a body of the type:
    // ```
    // {
    //   unsafe {
    //   asm!("NOP");
    //   }
    //   ::core::hint::black_box(primal(args));
    //   ::core::hint::black_box((args, ret));
    //   <This part remains to be done by following function>
    // }
    // ```
    fn init_body_helper(
        ecx: &ExtCtxt<'_>,
        span: Span,
        primal: Ident,
        new_names: &[String],
        sig_span: Span,
        new_decl_span: Span,
        idents: &[Ident],
        errored: bool,
        generics: &Generics,
    ) -> (P<ast::Block>, P<ast::Expr>, P<ast::Expr>, P<ast::Expr>) {
        let blackbox_path = ecx.std_path(&[sym::hint, sym::black_box]);
        let noop = ast::InlineAsm {
            asm_macro: ast::AsmMacro::Asm,
            template: vec![ast::InlineAsmTemplatePiece::String("NOP".into())],
            template_strs: Box::new([]),
            operands: vec![],
            clobber_abis: vec![],
            options: ast::InlineAsmOptions::PURE | ast::InlineAsmOptions::NOMEM,
            line_spans: vec![],
        };
        let noop_expr = ecx.expr_asm(span, P(noop));
        let unsf = ast::BlockCheckMode::Unsafe(ast::UnsafeSource::CompilerGenerated);
        let unsf_block = ast::Block {
            stmts: thin_vec![ecx.stmt_semi(noop_expr)],
            id: ast::DUMMY_NODE_ID,
            tokens: None,
            rules: unsf,
            span,
        };
        let unsf_expr = ecx.expr_block(P(unsf_block));
        let blackbox_call_expr = ecx.expr_path(ecx.path(span, blackbox_path));
        let primal_call = gen_primal_call(ecx, span, primal, idents, generics);
        let black_box_primal_call = ecx.expr_call(
            new_decl_span,
            blackbox_call_expr.clone(),
            thin_vec![primal_call.clone()],
        );
        let tup_args = new_names
            .iter()
            .map(|arg| ecx.expr_path(ecx.path_ident(span, Ident::from_str(arg))))
            .collect();

        let black_box_remaining_args = ecx.expr_call(
            sig_span,
            blackbox_call_expr.clone(),
            thin_vec![ecx.expr_tuple(sig_span, tup_args)],
        );

        let mut body = ecx.block(span, ThinVec::new());
        body.stmts.push(ecx.stmt_semi(unsf_expr));

        // This uses primal args which won't be available if we errored before
        if !errored {
            body.stmts.push(ecx.stmt_semi(black_box_primal_call.clone()));
        }
        body.stmts.push(ecx.stmt_semi(black_box_remaining_args));

        (body, primal_call, black_box_primal_call, blackbox_call_expr)
    }

    /// We only want this function to type-check, since we will replace the body
    /// later on llvm level. Using `loop {}` does not cover all return types anymore,
    /// so instead we manually build something that should pass the type checker.
    /// We also add a inline_asm line, as one more barrier for rustc to prevent inlining
    /// or const propagation. inline_asm will also triggers an Enzyme crash if due to another
    /// bug would ever try to accidentially differentiate this placeholder function body.
    /// Finally, we also add back_box usages of all input arguments, to prevent rustc
    /// from optimizing any arguments away.
    fn gen_enzyme_body(
        ecx: &ExtCtxt<'_>,
        x: &AutoDiffAttrs,
        n_active: u32,
        sig: &ast::FnSig,
        d_sig: &ast::FnSig,
        primal: Ident,
        new_names: &[String],
        span: Span,
        sig_span: Span,
        idents: Vec<Ident>,
        errored: bool,
        generics: &Generics,
    ) -> P<ast::Block> {
        let new_decl_span = d_sig.span;

        // Just adding some default inline-asm and black_box usages to prevent early inlining
        // and optimizations which alter the function signature.
        //
        // The bb_primal_call is the black_box call of the primal function. We keep it around,
        // since it has the convenient property of returning the type of the primal function,
        // Remember, we only care to match types here.
        // No matter which return we pick, we always wrap it into a std::hint::black_box call,
        // to prevent rustc from propagating it into the caller.
        let (mut body, primal_call, bb_primal_call, bb_call_expr) = init_body_helper(
            ecx,
            span,
            primal,
            new_names,
            sig_span,
            new_decl_span,
            &idents,
            errored,
            generics,
        );

        if !has_ret(&d_sig.decl.output) {
            // there is no return type that we have to match, () works fine.
            return body;
        }

        // Everything from here onwards just tries to fullfil the return type. Fun!

        // having an active-only return means we'll drop the original return type.
        // So that can be treated identical to not having one in the first place.
        let primal_ret = has_ret(&sig.decl.output) && !x.has_active_only_ret();

        if primal_ret && n_active == 0 && x.mode.is_rev() {
            // We only have the primal ret.
            body.stmts.push(ecx.stmt_expr(bb_primal_call));
            return body;
        }

        if !primal_ret && n_active == 1 {
            // Again no tuple return, so return default float val.
            let ty = match d_sig.decl.output {
                FnRetTy::Ty(ref ty) => ty.clone(),
                FnRetTy::Default(span) => {
                    panic!("Did not expect Default ret ty: {:?}", span);
                }
            };
            let arg = ty.kind.is_simple_path().unwrap();
            let tmp = ecx.def_site_path(&[arg, kw::Default]);
            let default_call_expr = ecx.expr_path(ecx.path(span, tmp));
            let default_call_expr = ecx.expr_call(new_decl_span, default_call_expr, thin_vec![]);
            body.stmts.push(ecx.stmt_expr(default_call_expr));
            return body;
        }

        let mut exprs: P<ast::Expr> = primal_call;
        let d_ret_ty = match d_sig.decl.output {
            FnRetTy::Ty(ref ty) => ty.clone(),
            FnRetTy::Default(span) => {
                panic!("Did not expect Default ret ty: {:?}", span);
            }
        };
        if x.mode.is_fwd() {
            // Fwd mode is easy. If the return activity is Const, we support arbitrary types.
            // Otherwise, we only support a scalar, a pair of scalars, or an array of scalars.
            // We checked that (on a best-effort base) in the preceding gen_enzyme_decl function.
            // In all three cases, we can return `std::hint::black_box(<T>::default())`.
            if x.ret_activity == DiffActivity::Const {
                // Here we call the primal function, since our dummy function has the same return
                // type due to the Const return activity.
                exprs = ecx.expr_call(new_decl_span, bb_call_expr, thin_vec![exprs]);
            } else {
                let q = QSelf { ty: d_ret_ty, path_span: span, position: 0 };
                let y = ExprKind::Path(
                    Some(P(q)),
                    ecx.path_ident(span, Ident::with_dummy_span(kw::Default)),
                );
                let default_call_expr = ecx.expr(span, y);
                let default_call_expr =
                    ecx.expr_call(new_decl_span, default_call_expr, thin_vec![]);
                exprs = ecx.expr_call(new_decl_span, bb_call_expr, thin_vec![default_call_expr]);
            }
        } else if x.mode.is_rev() {
            if x.width == 1 {
                // We either have `-> ArbitraryType` or `-> (ArbitraryType, repeated_float_scalars)`.
                match d_ret_ty.kind {
                    TyKind::Tup(ref args) => {
                        // We have a tuple return type. We need to create a tuple of the same size
                        // and fill it with default values.
                        let mut exprs2 = thin_vec![exprs];
                        for arg in args.iter().skip(1) {
                            let arg = arg.kind.is_simple_path().unwrap();
                            let tmp = ecx.def_site_path(&[arg, kw::Default]);
                            let default_call_expr = ecx.expr_path(ecx.path(span, tmp));
                            let default_call_expr =
                                ecx.expr_call(new_decl_span, default_call_expr, thin_vec![]);
                            exprs2.push(default_call_expr);
                        }
                        exprs = ecx.expr_tuple(new_decl_span, exprs2);
                    }
                    _ => {
                        // Interestingly, even the `-> ArbitraryType` case
                        // ends up getting matched and handled correctly above,
                        // so we don't have to handle any other case for now.
                        panic!("Unsupported return type: {:?}", d_ret_ty);
                    }
                }
            }
            exprs = ecx.expr_call(new_decl_span, bb_call_expr, thin_vec![exprs]);
        } else {
            unreachable!("Unsupported mode: {:?}", x.mode);
        }

        body.stmts.push(ecx.stmt_expr(exprs));

        body
    }

    fn gen_primal_call(
        ecx: &ExtCtxt<'_>,
        span: Span,
        primal: Ident,
        idents: &[Ident],
        generics: &Generics,
    ) -> P<ast::Expr> {
        let has_self = idents.len() > 0 && idents[0].name == kw::SelfLower;

        if has_self {
            let args: ThinVec<_> =
                idents[1..].iter().map(|arg| ecx.expr_path(ecx.path_ident(span, *arg))).collect();
            let self_expr = ecx.expr_self(span);
            ecx.expr_method_call(span, self_expr, primal, args)
        } else {
            let args: ThinVec<_> =
                idents.iter().map(|arg| ecx.expr_path(ecx.path_ident(span, *arg))).collect();
            let mut primal_path = ecx.path_ident(span, primal);

            let is_generic = !generics.params.is_empty();

            match (is_generic, primal_path.segments.last_mut()) {
                (true, Some(function_path)) => {
                    let primal_generic_types = generics
                        .params
                        .iter()
                        .filter(|param| matches!(param.kind, ast::GenericParamKind::Type { .. }));

                    let generated_generic_types = primal_generic_types
                        .map(|type_param| {
                            let generic_param = TyKind::Path(
                                None,
                                ast::Path {
                                    span,
                                    segments: thin_vec![ast::PathSegment {
                                        ident: type_param.ident,
                                        args: None,
                                        id: ast::DUMMY_NODE_ID,
                                    }],
                                    tokens: None,
                                },
                            );

                            ast::AngleBracketedArg::Arg(ast::GenericArg::Type(P(ast::Ty {
                                id: type_param.id,
                                span,
                                kind: generic_param,
                                tokens: None,
                            })))
                        })
                        .collect();

                    function_path.args =
                        Some(P(ast::GenericArgs::AngleBracketed(ast::AngleBracketedArgs {
                            span,
                            args: generated_generic_types,
                        })));
                }
                _ => {}
            }

            let primal_call_expr = ecx.expr_path(primal_path);
            ecx.expr_call(span, primal_call_expr, args)
        }
    }

    // Generate the new function declaration. Const arguments are kept as is. Duplicated arguments must
    // be pointers or references. Those receive a shadow argument, which is a mutable reference/pointer.
    // Active arguments must be scalars. Their shadow argument is added to the return type (and will be
    // zero-initialized by Enzyme).
    // Each argument of the primal function (and the return type if existing) must be annotated with an
    // activity.
    //
    // Error handling: If the user provides an invalid configuration (incorrect numbers, types, or
    // both), we emit an error and return the original signature. This allows us to continue parsing.
    // FIXME(Sa4dUs): make individual activities' span available so errors
    // can point to only the activity instead of the entire attribute
    fn gen_enzyme_decl(
        ecx: &ExtCtxt<'_>,
        sig: &ast::FnSig,
        x: &AutoDiffAttrs,
        span: Span,
    ) -> (ast::FnSig, Vec<String>, Vec<Ident>, bool) {
        let dcx = ecx.sess.dcx();
        let has_ret = has_ret(&sig.decl.output);
        let sig_args = sig.decl.inputs.len() + if has_ret { 1 } else { 0 };
        let num_activities = x.input_activity.len() + if x.has_ret_activity() { 1 } else { 0 };
        if sig_args != num_activities {
            dcx.emit_err(errors::AutoDiffInvalidNumberActivities {
                span,
                expected: sig_args,
                found: num_activities,
            });
            // This is not the right signature, but we can continue parsing.
            return (sig.clone(), vec![], vec![], true);
        }
        assert!(sig.decl.inputs.len() == x.input_activity.len());
        assert!(has_ret == x.has_ret_activity());
        let mut d_decl = sig.decl.clone();
        let mut d_inputs = Vec::new();
        let mut new_inputs = Vec::new();
        let mut idents = Vec::new();
        let mut act_ret = ThinVec::new();

        // We have two loops, a first one just to check the activities and types and possibly report
        // multiple errors in one compilation session.
        let mut errors = false;
        for (arg, activity) in sig.decl.inputs.iter().zip(x.input_activity.iter()) {
            if !valid_input_activity(x.mode, *activity) {
                dcx.emit_err(errors::AutoDiffInvalidApplicationModeAct {
                    span,
                    mode: x.mode.to_string(),
                    act: activity.to_string(),
                });
                errors = true;
            }
            if !valid_ty_for_activity(&arg.ty, *activity) {
                dcx.emit_err(errors::AutoDiffInvalidTypeForActivity {
                    span: arg.ty.span,
                    act: activity.to_string(),
                });
                errors = true;
            }
        }

        if has_ret && !valid_ret_activity(x.mode, x.ret_activity) {
            dcx.emit_err(errors::AutoDiffInvalidRetAct {
                span,
                mode: x.mode.to_string(),
                act: x.ret_activity.to_string(),
            });
            // We don't set `errors = true` to avoid annoying type errors relative
            // to the expanded macro type signature
        }

        if errors {
            // This is not the right signature, but we can continue parsing.
            return (sig.clone(), new_inputs, idents, true);
        }

        let unsafe_activities = x
            .input_activity
            .iter()
            .any(|&act| matches!(act, DiffActivity::DuplicatedOnly | DiffActivity::DualOnly));
        for (arg, activity) in sig.decl.inputs.iter().zip(x.input_activity.iter()) {
            d_inputs.push(arg.clone());
            match activity {
                DiffActivity::Active => {
                    act_ret.push(arg.ty.clone());
                    // if width =/= 1, then push [arg.ty; width] to act_ret
                }
                DiffActivity::ActiveOnly => {
                    // We will add the active scalar to the return type.
                    // This is handled later.
                }
                DiffActivity::Duplicated | DiffActivity::DuplicatedOnly => {
                    for i in 0..x.width {
                        let mut shadow_arg = arg.clone();
                        // We += into the shadow in reverse mode.
                        shadow_arg.ty = P(assure_mut_ref(&arg.ty));
                        let old_name = if let PatKind::Ident(_, ident, _) = arg.pat.kind {
                            ident.name
                        } else {
                            debug!("{:#?}", &shadow_arg.pat);
                            panic!("not an ident?");
                        };
                        let name: String = format!("d{}_{}", old_name, i);
                        new_inputs.push(name.clone());
                        let ident = Ident::from_str_and_span(&name, shadow_arg.pat.span);
                        shadow_arg.pat = P(ast::Pat {
                            id: ast::DUMMY_NODE_ID,
                            kind: PatKind::Ident(BindingMode::NONE, ident, None),
                            span: shadow_arg.pat.span,
                            tokens: shadow_arg.pat.tokens.clone(),
                        });
                        d_inputs.push(shadow_arg.clone());
                    }
                }
                DiffActivity::Dual
                | DiffActivity::DualOnly
                | DiffActivity::Dualv
                | DiffActivity::DualvOnly => {
                    // the *v variants get lowered to enzyme_dupv and enzyme_dupnoneedv, which cause
                    // Enzyme to not expect N arguments, but one argument (which is instead larger).
                    let iterations =
                        if matches!(activity, DiffActivity::Dualv | DiffActivity::DualvOnly) {
                            1
                        } else {
                            x.width
                        };
                    for i in 0..iterations {
                        let mut shadow_arg = arg.clone();
                        let old_name = if let PatKind::Ident(_, ident, _) = arg.pat.kind {
                            ident.name
                        } else {
                            debug!("{:#?}", &shadow_arg.pat);
                            panic!("not an ident?");
                        };
                        let name: String = format!("b{}_{}", old_name, i);
                        new_inputs.push(name.clone());
                        let ident = Ident::from_str_and_span(&name, shadow_arg.pat.span);
                        shadow_arg.pat = P(ast::Pat {
                            id: ast::DUMMY_NODE_ID,
                            kind: PatKind::Ident(BindingMode::NONE, ident, None),
                            span: shadow_arg.pat.span,
                            tokens: shadow_arg.pat.tokens.clone(),
                        });
                        d_inputs.push(shadow_arg.clone());
                    }
                }
                DiffActivity::Const => {
                    // Nothing to do here.
                }
                DiffActivity::None | DiffActivity::FakeActivitySize(_) => {
                    panic!("Should not happen");
                }
            }
            if let PatKind::Ident(_, ident, _) = arg.pat.kind {
                idents.push(ident.clone());
            } else {
                panic!("not an ident?");
            }
        }

        let active_only_ret = x.ret_activity == DiffActivity::ActiveOnly;
        if active_only_ret {
            assert!(x.mode.is_rev());
        }

        // If we return a scalar in the primal and the scalar is active,
        // then add it as last arg to the inputs.
        if x.mode.is_rev() {
            match x.ret_activity {
                DiffActivity::Active | DiffActivity::ActiveOnly => {
                    let ty = match d_decl.output {
                        FnRetTy::Ty(ref ty) => ty.clone(),
                        FnRetTy::Default(span) => {
                            panic!("Did not expect Default ret ty: {:?}", span);
                        }
                    };
                    let name = "dret".to_string();
                    let ident = Ident::from_str_and_span(&name, ty.span);
                    let shadow_arg = ast::Param {
                        attrs: ThinVec::new(),
                        ty: ty.clone(),
                        pat: P(ast::Pat {
                            id: ast::DUMMY_NODE_ID,
                            kind: PatKind::Ident(BindingMode::NONE, ident, None),
                            span: ty.span,
                            tokens: None,
                        }),
                        id: ast::DUMMY_NODE_ID,
                        span: ty.span,
                        is_placeholder: false,
                    };
                    d_inputs.push(shadow_arg);
                    new_inputs.push(name);
                }
                _ => {}
            }
        }
        d_decl.inputs = d_inputs.into();

        if x.mode.is_fwd() {
            let ty = match d_decl.output {
                FnRetTy::Ty(ref ty) => ty.clone(),
                FnRetTy::Default(span) => {
                    // We want to return std::hint::black_box(()).
                    let kind = TyKind::Tup(ThinVec::new());
                    let ty = P(rustc_ast::Ty { kind, id: ast::DUMMY_NODE_ID, span, tokens: None });
                    d_decl.output = FnRetTy::Ty(ty.clone());
                    assert!(matches!(x.ret_activity, DiffActivity::None));
                    // this won't be used below, so any type would be fine.
                    ty
                }
            };

            if matches!(x.ret_activity, DiffActivity::Dual | DiffActivity::Dualv) {
                let kind = if x.width == 1 || matches!(x.ret_activity, DiffActivity::Dualv) {
                    // Dual can only be used for f32/f64 ret.
                    // In that case we return now a tuple with two floats.
                    TyKind::Tup(thin_vec![ty.clone(), ty.clone()])
                } else {
                    // We have to return [T; width+1], +1 for the primal return.
                    let anon_const = rustc_ast::AnonConst {
                        id: ast::DUMMY_NODE_ID,
                        value: ecx.expr_usize(span, 1 + x.width as usize),
                    };
                    TyKind::Array(ty.clone(), anon_const)
                };
                let ty = P(rustc_ast::Ty { kind, id: ty.id, span: ty.span, tokens: None });
                d_decl.output = FnRetTy::Ty(ty);
            }
            if matches!(x.ret_activity, DiffActivity::DualOnly | DiffActivity::DualvOnly) {
                // No need to change the return type,
                // we will just return the shadow in place of the primal return.
                // However, if we have a width > 1, then we don't return -> T, but -> [T; width]
                if x.width > 1 {
                    let anon_const = rustc_ast::AnonConst {
                        id: ast::DUMMY_NODE_ID,
                        value: ecx.expr_usize(span, x.width as usize),
                    };
                    let kind = TyKind::Array(ty.clone(), anon_const);
                    let ty = P(rustc_ast::Ty { kind, id: ty.id, span: ty.span, tokens: None });
                    d_decl.output = FnRetTy::Ty(ty);
                }
            }
        }

        // If we use ActiveOnly, drop the original return value.
        d_decl.output =
            if active_only_ret { FnRetTy::Default(span) } else { d_decl.output.clone() };

        trace!("act_ret: {:?}", act_ret);

        // If we have an active input scalar, add it's gradient to the
        // return type. This might require changing the return type to a
        // tuple.
        if act_ret.len() > 0 {
            let ret_ty = match d_decl.output {
                FnRetTy::Ty(ref ty) => {
                    if !active_only_ret {
                        act_ret.insert(0, ty.clone());
                    }
                    let kind = TyKind::Tup(act_ret);
                    P(rustc_ast::Ty { kind, id: ty.id, span: ty.span, tokens: None })
                }
                FnRetTy::Default(span) => {
                    if act_ret.len() == 1 {
                        act_ret[0].clone()
                    } else {
                        let kind = TyKind::Tup(act_ret.iter().map(|arg| arg.clone()).collect());
                        P(rustc_ast::Ty { kind, id: ast::DUMMY_NODE_ID, span, tokens: None })
                    }
                }
            };
            d_decl.output = FnRetTy::Ty(ret_ty);
        }

        let mut d_header = sig.header.clone();
        if unsafe_activities {
            d_header.safety = rustc_ast::Safety::Unsafe(span);
        }
        let d_sig = FnSig { header: d_header, decl: d_decl, span };
        trace!("Generated signature: {:?}", d_sig);
        (d_sig, new_inputs, idents, false)
    }
}

pub(crate) use llvm_enzyme::{expand_forward, expand_reverse};
