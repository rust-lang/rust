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
    use rustc_ast::token::{Lit, LitKind, Token, TokenKind};
    use rustc_ast::tokenstream::*;
    use rustc_ast::visit::AssocCtxt::*;
    use rustc_ast::{
        self as ast, AngleBracketedArg, AngleBracketedArgs, AnonConst, AssocItemKind, BindingMode,
        FnRetTy, FnSig, GenericArg, GenericArgs, GenericParamKind, Generics, ItemKind,
        MetaItemInner, PatKind, Path, PathSegment, TyKind, Visibility,
    };
    use rustc_expand::base::{Annotatable, ExtCtxt};
    use rustc_span::{Ident, Span, Symbol, sym};
    use thin_vec::{ThinVec, thin_vec};
    use tracing::{debug, trace};

    use crate::errors;

    pub(crate) fn outer_normal_attr(
        kind: &Box<rustc_ast::NormalAttr>,
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
    fn extract_item_info(iitem: &Box<ast::Item>) -> Option<(Visibility, FnSig, Ident, Generics)> {
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
    /// type-checking and can be called by users. The exact signature of the generated function
    /// depends on the configuration provided by the user, but here is an example:
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
    /// fn sin(x: &Box<f32>) -> f32 {
    ///     f32::sin(**x)
    /// }
    /// #[rustc_autodiff(Reverse, Duplicated, Active)]
    /// fn cos_box(x: &Box<f32>, dx: &mut Box<f32>, dret: f32) -> f32 {
    ///     std::intrinsics::autodiff(sin::<>, cos_box::<>, (x, dx, dret))
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
        // FIXME(bjorn3) maybe have the backend directly tell if autodiff is supported?
        if cfg!(not(feature = "llvm_enzyme")) {
            ecx.sess.dcx().emit_err(errors::AutoDiffSupportNotBuild { span: meta_item.span });
            return vec![item];
        }
        let dcx = ecx.sess.dcx();

        // first get information about the annotable item: visibility, signature, name and generic
        // parameters.
        // these will be used to generate the differentiated version of the function
        let Some((vis, sig, primal, generics, impl_of_trait)) = (match &item {
            Annotatable::Item(iitem) => {
                extract_item_info(iitem).map(|(v, s, p, g)| (v, s, p, g, false))
            }
            Annotatable::Stmt(stmt) => match &stmt.kind {
                ast::StmtKind::Item(iitem) => {
                    extract_item_info(iitem).map(|(v, s, p, g)| (v, s, p, g, false))
                }
                _ => None,
            },
            Annotatable::AssocItem(assoc_item, Impl { of_trait }) => match &assoc_item.kind {
                ast::AssocItemKind::Fn(box ast::Fn { sig, ident, generics, .. }) => Some((
                    assoc_item.vis.clone(),
                    sig.clone(),
                    ident.clone(),
                    generics.clone(),
                    *of_trait,
                )),
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

        let d_sig = gen_enzyme_decl(ecx, &sig, &x, span);

        let d_body = ecx.block(
            span,
            thin_vec![call_autodiff(
                ecx,
                primal,
                first_ident(&meta_item_vec[0]),
                span,
                &d_sig,
                &generics,
                impl_of_trait,
            )],
        );

        // The first element of it is the name of the function to be generated
        let d_fn = Box::new(ast::Fn {
            defaultness: ast::Defaultness::Final,
            sig: d_sig,
            ident: first_ident(&meta_item_vec[0]),
            generics,
            contract: None,
            body: Some(d_body),
            define_opaque: None,
        });
        let mut rustc_ad_attr =
            Box::new(ast::NormalAttr::from_ident(Ident::with_dummy_span(sym::rustc_autodiff)));

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
        let inline_never_attr = Box::new(ast::NormalAttr { item: inline_item, tokens: None });
        let new_id = ecx.sess.psess.attr_id_generator.mk_attr_id();
        let attr = outer_normal_attr(&rustc_ad_attr, new_id, span);
        let new_id = ecx.sess.psess.attr_id_generator.mk_attr_id();
        let inline_never = outer_normal_attr(&inline_never_attr, new_id, span);

        // We're avoid duplicating the attribute `#[rustc_autodiff]`.
        fn same_attribute(attr: &ast::AttrKind, item: &ast::AttrKind) -> bool {
            match (attr, item) {
                (ast::AttrKind::Normal(a), ast::AttrKind::Normal(b)) => {
                    let a = &a.item.path;
                    let b = &b.item.path;
                    a.segments.iter().eq_by(&b.segments, |a, b| a.ident == b.ident)
                }
                _ => false,
            }
        }

        let mut has_inline_never = false;

        // Don't add it multiple times:
        let orig_annotatable: Annotatable = match item {
            Annotatable::Item(ref mut iitem) => {
                if !iitem.attrs.iter().any(|a| same_attribute(&a.kind, &attr.kind)) {
                    iitem.attrs.push(attr);
                }
                if iitem.attrs.iter().any(|a| same_attribute(&a.kind, &inline_never.kind)) {
                    has_inline_never = true;
                }
                Annotatable::Item(iitem.clone())
            }
            Annotatable::AssocItem(ref mut assoc_item, i @ Impl { .. }) => {
                if !assoc_item.attrs.iter().any(|a| same_attribute(&a.kind, &attr.kind)) {
                    assoc_item.attrs.push(attr);
                }
                if assoc_item.attrs.iter().any(|a| same_attribute(&a.kind, &inline_never.kind)) {
                    has_inline_never = true;
                }
                Annotatable::AssocItem(assoc_item.clone(), i)
            }
            Annotatable::Stmt(ref mut stmt) => {
                match stmt.kind {
                    ast::StmtKind::Item(ref mut iitem) => {
                        if !iitem.attrs.iter().any(|a| same_attribute(&a.kind, &attr.kind)) {
                            iitem.attrs.push(attr);
                        }
                        if iitem.attrs.iter().any(|a| same_attribute(&a.kind, &inline_never.kind)) {
                            has_inline_never = true;
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

        let new_id = ecx.sess.psess.attr_id_generator.mk_attr_id();
        let d_attr = outer_normal_attr(&rustc_ad_attr, new_id, span);

        // If the source function has the `#[inline(never)]` attribute, we'll also add it to the diff function
        let mut d_attrs = thin_vec![d_attr];

        if has_inline_never {
            d_attrs.push(inline_never);
        }

        let d_annotatable = match &item {
            Annotatable::AssocItem(_, _) => {
                let assoc_item: AssocItemKind = ast::AssocItemKind::Fn(d_fn);
                let d_fn = Box::new(ast::AssocItem {
                    attrs: d_attrs,
                    id: ast::DUMMY_NODE_ID,
                    span,
                    vis,
                    kind: assoc_item,
                    tokens: None,
                });
                Annotatable::AssocItem(d_fn, Impl { of_trait: false })
            }
            Annotatable::Item(_) => {
                let mut d_fn = ecx.item(span, d_attrs, ItemKind::Fn(d_fn));
                d_fn.vis = vis;

                Annotatable::Item(d_fn)
            }
            Annotatable::Stmt(_) => {
                let mut d_fn = ecx.item(span, d_attrs, ItemKind::Fn(d_fn));
                d_fn.vis = vis;

                Annotatable::Stmt(Box::new(ast::Stmt {
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

    // Generate `autodiff` intrinsic call
    // ```
    // std::intrinsics::autodiff(source, diff, (args))
    // ```
    fn call_autodiff(
        ecx: &ExtCtxt<'_>,
        primal: Ident,
        diff: Ident,
        span: Span,
        d_sig: &FnSig,
        generics: &Generics,
        is_impl: bool,
    ) -> rustc_ast::Stmt {
        let primal_path_expr = gen_turbofish_expr(ecx, primal, generics, span, is_impl);
        let diff_path_expr = gen_turbofish_expr(ecx, diff, generics, span, is_impl);

        let tuple_expr = ecx.expr_tuple(
            span,
            d_sig
                .decl
                .inputs
                .iter()
                .map(|arg| match arg.pat.kind {
                    PatKind::Ident(_, ident, _) => ecx.expr_path(ecx.path_ident(span, ident)),
                    _ => todo!(),
                })
                .collect::<ThinVec<_>>()
                .into(),
        );

        let enzyme_path_idents = ecx.std_path(&[sym::intrinsics, sym::autodiff]);
        let enzyme_path = ecx.path(span, enzyme_path_idents);
        let call_expr = ecx.expr_call(
            span,
            ecx.expr_path(enzyme_path),
            vec![primal_path_expr, diff_path_expr, tuple_expr].into(),
        );

        ecx.stmt_expr(call_expr)
    }

    // Generate turbofish expression from fn name and generics
    // Given `foo` and `<A, B, C>` params, gen `foo::<A, B, C>`
    // We use this expression when passing primal and diff function to the autodiff intrinsic
    fn gen_turbofish_expr(
        ecx: &ExtCtxt<'_>,
        ident: Ident,
        generics: &Generics,
        span: Span,
        is_impl: bool,
    ) -> Box<ast::Expr> {
        let generic_args = generics
            .params
            .iter()
            .filter_map(|p| match &p.kind {
                GenericParamKind::Type { .. } => {
                    let path = ast::Path::from_ident(p.ident);
                    let ty = ecx.ty_path(path);
                    Some(AngleBracketedArg::Arg(GenericArg::Type(ty)))
                }
                GenericParamKind::Const { .. } => {
                    let expr = ecx.expr_path(ast::Path::from_ident(p.ident));
                    let anon_const = AnonConst { id: ast::DUMMY_NODE_ID, value: expr };
                    Some(AngleBracketedArg::Arg(GenericArg::Const(anon_const)))
                }
                GenericParamKind::Lifetime { .. } => None,
            })
            .collect::<ThinVec<_>>();

        let args: AngleBracketedArgs = AngleBracketedArgs { span, args: generic_args };

        let segment = PathSegment {
            ident,
            id: ast::DUMMY_NODE_ID,
            args: Some(Box::new(GenericArgs::AngleBracketed(args))),
        };

        let segments = if is_impl {
            thin_vec![
                PathSegment { ident: Ident::from_str("Self"), id: ast::DUMMY_NODE_ID, args: None },
                segment,
            ]
        } else {
            thin_vec![segment]
        };

        let path = Path { span, segments, tokens: None };

        ecx.expr_path(path)
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
    ) -> ast::FnSig {
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
            return sig.clone();
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
            return sig.clone();
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
                        shadow_arg.ty = Box::new(assure_mut_ref(&arg.ty));
                        let old_name = if let PatKind::Ident(_, ident, _) = arg.pat.kind {
                            ident.name
                        } else {
                            debug!("{:#?}", &shadow_arg.pat);
                            panic!("not an ident?");
                        };
                        let name: String = format!("d{}_{}", old_name, i);
                        new_inputs.push(name.clone());
                        let ident = Ident::from_str_and_span(&name, shadow_arg.pat.span);
                        shadow_arg.pat = Box::new(ast::Pat {
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
                        shadow_arg.pat = Box::new(ast::Pat {
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
                        pat: Box::new(ast::Pat {
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
                    let ty = Box::new(rustc_ast::Ty {
                        kind,
                        id: ast::DUMMY_NODE_ID,
                        span,
                        tokens: None,
                    });
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
                let ty = Box::new(rustc_ast::Ty { kind, id: ty.id, span: ty.span, tokens: None });
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
                    let ty =
                        Box::new(rustc_ast::Ty { kind, id: ty.id, span: ty.span, tokens: None });
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
                    Box::new(rustc_ast::Ty { kind, id: ty.id, span: ty.span, tokens: None })
                }
                FnRetTy::Default(span) => {
                    if act_ret.len() == 1 {
                        act_ret[0].clone()
                    } else {
                        let kind = TyKind::Tup(act_ret.iter().map(|arg| arg.clone()).collect());
                        Box::new(rustc_ast::Ty { kind, id: ast::DUMMY_NODE_ID, span, tokens: None })
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
        d_sig
    }
}

pub(crate) use llvm_enzyme::{expand_forward, expand_reverse};
