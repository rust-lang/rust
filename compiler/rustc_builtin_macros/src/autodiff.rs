#![allow(unused_imports)]
use std::str::FromStr;
use std::string::String;

use rustc_ast::ast::Generics;
use rustc_ast::expand::autodiff_attrs::{
    is_fwd, is_rev, valid_input_activity, valid_ty_for_activity, AutoDiffAttrs, DiffActivity,
    DiffMode,
};
use rustc_ast::ptr::P;
use rustc_ast::token::{Token, TokenKind};
use rustc_ast::tokenstream::*;
use rustc_ast::{self as ast, NestedMetaItem};
use rustc_ast::{AssocItemKind, FnRetTy, FnSig, ItemKind, PatKind, TyKind};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::symbol::{kw, sym, Ident};
use rustc_span::{Span, Symbol};
use thin_vec::{thin_vec, ThinVec};
use tracing::trace;

use crate::errors;

#[cfg(not(llvm_enzyme))]
pub fn expand(
    ecx: &mut ExtCtxt<'_>,
    _expand_span: Span,
    meta_item: &ast::MetaItem,
    item: Annotatable,
) -> Vec<Annotatable> {
    ecx.sess.dcx().emit_err(errors::AutoDiffSupportNotBuild { span: meta_item.span });
    return vec![item];
}

#[cfg(llvm_enzyme)]
fn first_ident(x: &NestedMetaItem) -> rustc_span::symbol::Ident {
    let segments = &x.meta_item().unwrap().path.segments;
    assert!(segments.len() == 1);
    segments[0].ident
}

#[cfg(llvm_enzyme)]
fn name(x: &NestedMetaItem) -> String {
    first_ident(x).name.to_string()
}

#[cfg(llvm_enzyme)]
pub fn from_ast(
    ecx: &mut ExtCtxt<'_>,
    meta_item: &ThinVec<NestedMetaItem>,
    has_ret: bool,
) -> AutoDiffAttrs {
    let mode = name(&meta_item[1]);
    let mode = match DiffMode::from_str(&mode) {
        Ok(x) => x,
        Err(_) => {
            ecx.sess
                .dcx()
                .emit_err(errors::AutoDiffInvalidMode { span: meta_item[1].span(), mode });
            return AutoDiffAttrs::inactive();
        }
    };
    let mut activities: Vec<DiffActivity> = vec![];
    for x in &meta_item[2..] {
        let activity_str = name(&x);
        let res = DiffActivity::from_str(&activity_str);
        match res {
            Ok(x) => activities.push(x),
            Err(_) => {
                ecx.sess.dcx().emit_err(errors::AutoDiffUnknownActivity {
                    span: x.span(),
                    act: activity_str,
                });
                return AutoDiffAttrs::inactive();
            }
        };
    }

    // If a return type exist, we need to split the last activity,
    // otherwise we return None as placeholder.
    let (ret_activity, input_activity) = if has_ret {
        activities.split_last().unwrap()
    } else {
        (&DiffActivity::None, activities.as_slice())
    };

    AutoDiffAttrs { mode, ret_activity: *ret_activity, input_activity: input_activity.to_vec() }
}

#[cfg(llvm_enzyme)]
pub fn expand(
    ecx: &mut ExtCtxt<'_>,
    expand_span: Span,
    meta_item: &ast::MetaItem,
    mut item: Annotatable,
) -> Vec<Annotatable> {
    //check_builtin_macro_attribute(ecx, meta_item, sym::alloc_error_handler);

    // first get the annotable item:

    use ast::visit::AssocCtxt;
    let (sig, is_impl): (FnSig, bool) = match &item {
        Annotatable::Item(ref iitem) => {
            let sig = match &iitem.kind {
                ItemKind::Fn(box ast::Fn { sig, .. }) => sig,
                _ => {
                    ecx.sess
                        .dcx()
                        .emit_err(errors::AutoDiffInvalidApplication { span: item.span() });
                    return vec![item];
                }
            };
            (sig.clone(), false)
        }
        Annotatable::AssocItem(ref assoc_item, AssocCtxt::Impl) => {
            let sig = match &assoc_item.kind {
                ast::AssocItemKind::Fn(box ast::Fn { sig, .. }) => sig,
                _ => {
                    ecx.sess
                        .dcx()
                        .emit_err(errors::AutoDiffInvalidApplication { span: item.span() });
                    return vec![item];
                }
            };
            (sig.clone(), true)
        }
        _ => {
            ecx.sess.dcx().emit_err(errors::AutoDiffInvalidApplication { span: item.span() });
            return vec![item];
        }
    };

    let meta_item_vec: ThinVec<NestedMetaItem> = match meta_item.kind {
        ast::MetaItemKind::List(ref vec) => vec.clone(),
        _ => {
            ecx.sess.dcx().emit_err(errors::AutoDiffInvalidApplication { span: item.span() });
            return vec![item];
        }
    };

    let has_ret = sig.decl.output.has_ret();
    let sig_span = ecx.with_call_site_ctxt(sig.span);

    let (vis, primal) = match &item {
        Annotatable::Item(ref iitem) => (iitem.vis.clone(), iitem.ident.clone()),
        Annotatable::AssocItem(ref assoc_item, AssocCtxt::Impl) => {
            (assoc_item.vis.clone(), assoc_item.ident.clone())
        }
        _ => {
            ecx.sess.dcx().emit_err(errors::AutoDiffInvalidApplication { span: item.span() });
            return vec![item];
        }
    };

    // create TokenStream from vec elemtents:
    // meta_item doesn't have a .tokens field
    let comma: Token = Token::new(TokenKind::Comma, Span::default());
    let mut ts: Vec<TokenTree> = vec![];
    for t in meta_item_vec.clone()[1..].iter() {
        let val = first_ident(t);
        let t = Token::from_ast_ident(val);
        ts.push(TokenTree::Token(t, Spacing::Joint));
        ts.push(TokenTree::Token(comma.clone(), Spacing::Alone));
    }
    if !sig.decl.output.has_ret() {
        // We don't want users to provide a return activity if the function doesn't return anything.
        // For simplicity, we just add a dummy token to the end of the list.
        let t = Token::new(TokenKind::Ident(sym::None, false.into()), Span::default());
        ts.push(TokenTree::Token(t, Spacing::Joint));
    }
    let ts: TokenStream = TokenStream::from_iter(ts);

    let x: AutoDiffAttrs = from_ast(ecx, &meta_item_vec, has_ret);
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
    let (d_sig, new_args, idents) = gen_enzyme_decl(ecx, &sig, &x, span);
    let new_decl_span = d_sig.span;
    let d_body = gen_enzyme_body(
        ecx,
        &x,
        n_active,
        &sig,
        &d_sig,
        primal,
        &new_args,
        span,
        sig_span,
        new_decl_span,
        idents,
    );
    let d_ident = first_ident(&meta_item_vec[0]);

    // The first element of it is the name of the function to be generated
    let asdf = Box::new(ast::Fn {
        defaultness: ast::Defaultness::Final,
        sig: d_sig,
        generics: Generics::default(),
        body: Some(d_body),
    });
    let mut rustc_ad_attr =
        P(ast::NormalAttr::from_ident(Ident::with_dummy_span(sym::rustc_autodiff)));
    let ts2: Vec<TokenTree> = vec![TokenTree::Token(
        Token::new(TokenKind::Ident(sym::never, false.into()), span),
        Spacing::Joint,
    )];
    let never_arg = ast::DelimArgs {
        dspan: ast::tokenstream::DelimSpan::from_single(span),
        delim: ast::token::Delimiter::Parenthesis,
        tokens: ast::tokenstream::TokenStream::from_iter(ts2),
    };
    let inline_item = ast::AttrItem {
        unsafety: ast::Safety::Default,
        path: ast::Path::from_ident(Ident::with_dummy_span(sym::inline)),
        args: ast::AttrArgs::Delimited(never_arg),
        tokens: None,
    };
    let inline_never_attr = P(ast::NormalAttr { item: inline_item, tokens: None });
    let mut attr: ast::Attribute = ast::Attribute {
        kind: ast::AttrKind::Normal(rustc_ad_attr.clone()),
        //id: ast::DUMMY_TR_ID,
        id: ast::AttrId::from_u32(12341), // TODO: fix
        style: ast::AttrStyle::Outer,
        span,
    };
    let inline_never: ast::Attribute = ast::Attribute {
        kind: ast::AttrKind::Normal(inline_never_attr),
        //id: ast::DUMMY_TR_ID,
        id: ast::AttrId::from_u32(12342), // TODO: fix
        style: ast::AttrStyle::Outer,
        span,
    };

    // Don't add it multiple times:
    let orig_annotatable: Annotatable = match item {
        Annotatable::Item(ref mut iitem) => {
            if !iitem.attrs.iter().any(|a| a.id == attr.id) {
                iitem.attrs.push(attr.clone());
            }
            if !iitem.attrs.iter().any(|a| a.id == inline_never.id) {
                iitem.attrs.push(inline_never.clone());
            }
            Annotatable::Item(iitem.clone())
        }
        Annotatable::AssocItem(ref mut assoc_item, i @ AssocCtxt::Impl) => {
            if !assoc_item.attrs.iter().any(|a| a.id == attr.id) {
                assoc_item.attrs.push(attr.clone());
            }
            if !assoc_item.attrs.iter().any(|a| a.id == inline_never.id) {
                assoc_item.attrs.push(inline_never.clone());
            }
            Annotatable::AssocItem(assoc_item.clone(), i)
        }
        _ => {
            panic!("not supported");
        }
    };

    // Now update for d_fn
    rustc_ad_attr.item.args = rustc_ast::AttrArgs::Delimited(rustc_ast::DelimArgs {
        dspan: DelimSpan::dummy(),
        delim: rustc_ast::token::Delimiter::Parenthesis,
        tokens: ts,
    });
    attr.kind = ast::AttrKind::Normal(rustc_ad_attr);

    let d_annotatable = if is_impl {
        let assoc_item: AssocItemKind = ast::AssocItemKind::Fn(asdf);
        let d_fn = P(ast::AssocItem {
            attrs: thin_vec![attr.clone(), inline_never],
            id: ast::DUMMY_NODE_ID,
            span,
            vis,
            ident: d_ident,
            kind: assoc_item,
            tokens: None,
        });
        Annotatable::AssocItem(d_fn, AssocCtxt::Impl)
    } else {
        let mut d_fn =
            ecx.item(span, d_ident, thin_vec![attr.clone(), inline_never], ItemKind::Fn(asdf));
        d_fn.vis = vis;
        Annotatable::Item(d_fn)
    };

    return vec![orig_annotatable, d_annotatable];
}

// shadow arguments in reverse mode must be mutable references or ptrs, because Enzyme will write into them.
#[cfg(llvm_enzyme)]
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

// The body of our generated functions will consist of two black_Box calls.
// The first will call the primal function with the original arguments.
// The second will just take a tuple containing the new arguments.
// This way we surpress rustc from optimizing any argument away.
// The last line will 'loop {}', to match the return type of the new function
#[cfg(llvm_enzyme)]
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
    new_decl_span: Span,
    idents: Vec<Ident>,
) -> P<ast::Block> {
    let blackbox_path = ecx.std_path(&[Symbol::intern("hint"), Symbol::intern("black_box")]);
    let empty_loop_block = ecx.block(span, ThinVec::new());
    let noop = ast::InlineAsm {
        template: vec![ast::InlineAsmTemplatePiece::String("NOP".to_string().into())],
        template_strs: Box::new([]),
        operands: vec![],
        clobber_abis: vec![],
        options: ast::InlineAsmOptions::PURE & ast::InlineAsmOptions::NOMEM,
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
        could_be_bare_literal: false,
    };
    let unsf_expr = ecx.expr_block(P(unsf_block));
    let _loop_expr = ecx.expr_loop(span, empty_loop_block);
    let blackbox_call_expr = ecx.expr_path(ecx.path(span, blackbox_path));
    let primal_call = gen_primal_call(ecx, span, primal, idents);
    let black_box_primal_call =
        ecx.expr_call(new_decl_span, blackbox_call_expr.clone(), thin_vec![primal_call.clone()]);
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
    body.stmts.push(ecx.stmt_semi(black_box_primal_call.clone()));
    body.stmts.push(ecx.stmt_semi(black_box_remaining_args));

    if !d_sig.decl.output.has_ret() {
        // there is no return type that we have to match, () works fine.
        return body;
    }

    // having an active-only return means we'll drop the original return type.
    // So that can be treated identical to not having one in the first place.
    let primal_ret = sig.decl.output.has_ret() && !x.has_active_only_ret();

    if primal_ret && n_active == 0 && is_rev(x.mode) {
        // We only have the primal ret.
        body.stmts.push(ecx.stmt_expr(black_box_primal_call.clone()));
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
        let sl: Vec<Symbol> = vec![arg, Symbol::intern("default")];
        let tmp = ecx.def_site_path(&sl);
        let default_call_expr = ecx.expr_path(ecx.path(span, tmp));
        let default_call_expr = ecx.expr_call(new_decl_span, default_call_expr, thin_vec![]);
        body.stmts.push(ecx.stmt_expr(default_call_expr));
        return body;
    }

    let mut exprs = ThinVec::<P<ast::Expr>>::new();
    if primal_ret {
        // We have both primal ret and active floats.
        // primal ret is first, by construction.
        exprs.push(primal_call.clone());
    }

    // Now construct default placeholder for each active float.
    // Is there something nicer than f32::default() and f64::default()?
    let d_ret_ty = match d_sig.decl.output {
        FnRetTy::Ty(ref ty) => ty.clone(),
        FnRetTy::Default(span) => {
            panic!("Did not expect Default ret ty: {:?}", span);
        }
    };
    let mut d_ret_ty = match d_ret_ty.kind.clone() {
        TyKind::Tup(ref tys) => tys.clone(),
        TyKind::Path(_, rustc_ast::Path { segments, .. }) => {
            if segments.len() == 1 && segments[0].args.is_none() {
                let id = vec![segments[0].ident];
                let kind = TyKind::Path(None, ecx.path(span, id));
                let ty = P(rustc_ast::Ty { kind, id: ast::DUMMY_NODE_ID, span, tokens: None });
                thin_vec![ty]
            } else {
                panic!("Expected tuple or simple path return type");
            }
        }
        _ => {
            // We messed up construction of d_sig
            panic!("Did not expect non-tuple ret ty: {:?}", d_ret_ty);
        }
    };
    if is_fwd(x.mode) {
        if x.ret_activity == DiffActivity::Dual {
            assert!(d_ret_ty.len() == 2);
            // both should be identical, by construction
            let arg = d_ret_ty[0].kind.is_simple_path().unwrap();
            let arg2 = d_ret_ty[1].kind.is_simple_path().unwrap();
            assert!(arg == arg2);
            let sl: Vec<Symbol> = vec![arg, Symbol::intern("default")];
            let tmp = ecx.def_site_path(&sl);
            let default_call_expr = ecx.expr_path(ecx.path(span, tmp));
            let default_call_expr = ecx.expr_call(new_decl_span, default_call_expr, thin_vec![]);
            exprs.push(default_call_expr);
        }
    } else {
        assert!(is_rev(x.mode));

        if primal_ret {
            // We have extra handling above for the primal ret
            d_ret_ty = d_ret_ty[1..].to_vec().into();
        }

        for arg in d_ret_ty.iter() {
            let arg = arg.kind.is_simple_path().unwrap();
            let sl: Vec<Symbol> = vec![arg, Symbol::intern("default")];
            let tmp = ecx.def_site_path(&sl);
            let default_call_expr = ecx.expr_path(ecx.path(span, tmp));
            let default_call_expr = ecx.expr_call(new_decl_span, default_call_expr, thin_vec![]);
            exprs.push(default_call_expr);
        }
    }

    let ret: P<ast::Expr>;
    if exprs.len() > 1 {
        let ret_tuple: P<ast::Expr> = ecx.expr_tuple(span, exprs);
        ret = ecx.expr_call(new_decl_span, blackbox_call_expr.clone(), thin_vec![ret_tuple]);
    } else if exprs.len() == 1 {
        let ret_scal = exprs.pop().unwrap();
        ret = ecx.expr_call(new_decl_span, blackbox_call_expr.clone(), thin_vec![ret_scal]);
    } else {
        assert!(!d_sig.decl.output.has_ret());
        // We don't have to match the return type.
        return body;
    }
    assert!(d_sig.decl.output.has_ret());
    body.stmts.push(ecx.stmt_expr(ret));

    body
}

#[cfg(llvm_enzyme)]
fn gen_primal_call(
    ecx: &ExtCtxt<'_>,
    span: Span,
    primal: Ident,
    idents: Vec<Ident>,
) -> P<ast::Expr> {
    let has_self = idents.len() > 0 && idents[0].name == kw::SelfLower;
    if has_self {
        let args: ThinVec<_> =
            idents[1..].iter().map(|arg| ecx.expr_path(ecx.path_ident(span, *arg))).collect();
        let self_expr = ecx.expr_self(span);
        ecx.expr_method_call(span, self_expr, primal, args.clone())
    } else {
        let args: ThinVec<_> =
            idents.iter().map(|arg| ecx.expr_path(ecx.path_ident(span, *arg))).collect();
        let primal_call_expr = ecx.expr_path(ecx.path_ident(span, primal));
        ecx.expr_call(span, primal_call_expr, args)
    }
}

// Generate the new function declaration. Const arguments are kept as is. Duplicated arguments must
// be pointers or references. Those receive a shadow argument, which is a mutable reference/pointer.
// Active arguments must be scalars. Their shadow argument is added to the return type (and will be
// zero-initialized by Enzyme). Active arguments are not handled yet.
// Each argument of the primal function (and the return type if existing) must be annotated with an
// activity.
#[cfg(llvm_enzyme)]
fn gen_enzyme_decl(
    ecx: &ExtCtxt<'_>,
    sig: &ast::FnSig,
    x: &AutoDiffAttrs,
    span: Span,
) -> (ast::FnSig, Vec<String>, Vec<Ident>) {
    use ast::BindingMode;

    let sig_args = sig.decl.inputs.len() + if sig.decl.output.has_ret() { 1 } else { 0 };
    let num_activities = x.input_activity.len() + if x.has_ret_activity() { 1 } else { 0 };
    if sig_args != num_activities {
        ecx.sess.dcx().emit_fatal(errors::AutoDiffInvalidNumberActivities {
            span,
            expected: sig_args,
            found: num_activities,
        });
    }
    assert!(sig.decl.inputs.len() == x.input_activity.len());
    assert!(sig.decl.output.has_ret() == x.has_ret_activity());
    let mut d_decl = sig.decl.clone();
    let mut d_inputs = Vec::new();
    let mut new_inputs = Vec::new();
    let mut idents = Vec::new();
    let mut act_ret = ThinVec::new();
    for (arg, activity) in sig.decl.inputs.iter().zip(x.input_activity.iter()) {
        d_inputs.push(arg.clone());
        if !valid_input_activity(x.mode, *activity) {
            ecx.sess.dcx().emit_err(errors::AutoDiffInvalidApplicationModeAct {
                span,
                mode: x.mode.to_string(),
                act: activity.to_string(),
            });
        }
        if !valid_ty_for_activity(&arg.ty, *activity) {
            ecx.sess.dcx().emit_err(errors::AutoDiffInvalidTypeForActivity {
                span: arg.ty.span,
                act: activity.to_string(),
            });
        }
        match activity {
            DiffActivity::Active => {
                act_ret.push(arg.ty.clone());
            }
            DiffActivity::Duplicated => {
                let mut shadow_arg = arg.clone();
                // We += into the shadow in reverse mode.
                shadow_arg.ty = P(assure_mut_ref(&arg.ty));
                let old_name = if let PatKind::Ident(_, ident, _) = arg.pat.kind {
                    ident.name
                } else {
                    dbg!(&shadow_arg.pat);
                    panic!("not an ident?");
                };
                let name: String = format!("d{}", old_name);
                new_inputs.push(name.clone());
                let ident = Ident::from_str_and_span(&name, shadow_arg.pat.span);
                shadow_arg.pat = P(ast::Pat {
                    // TODO: Check id
                    id: ast::DUMMY_NODE_ID,
                    kind: PatKind::Ident(BindingMode::NONE, ident, None),
                    span: shadow_arg.pat.span,
                    tokens: shadow_arg.pat.tokens.clone(),
                });
                d_inputs.push(shadow_arg);
            }
            DiffActivity::Dual => {
                let mut shadow_arg = arg.clone();
                let old_name = if let PatKind::Ident(_, ident, _) = arg.pat.kind {
                    ident.name
                } else {
                    dbg!(&shadow_arg.pat);
                    panic!("not an ident?");
                };
                let name: String = format!("b{}", old_name);
                new_inputs.push(name.clone());
                let ident = Ident::from_str_and_span(&name, shadow_arg.pat.span);
                shadow_arg.pat = P(ast::Pat {
                    // TODO: Check id
                    id: ast::DUMMY_NODE_ID,
                    kind: PatKind::Ident(BindingMode::NONE, ident, None),
                    span: shadow_arg.pat.span,
                    tokens: shadow_arg.pat.tokens.clone(),
                });
                d_inputs.push(shadow_arg);
            }
            DiffActivity::Const => {
                // Nothing to do here.
            }
            _ => {
                dbg!(&activity);
                panic!("Not implemented");
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
        assert!(is_rev(x.mode));
    }

    // If we return a scalar in the primal and the scalar is active,
    // then add it as last arg to the inputs.
    if is_rev(x.mode) {
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

    if is_fwd(x.mode) {
        if let DiffActivity::Dual = x.ret_activity {
            let ty = match d_decl.output {
                FnRetTy::Ty(ref ty) => ty.clone(),
                FnRetTy::Default(span) => {
                    panic!("Did not expect Default ret ty: {:?}", span);
                }
            };
            // Dual can only be used for f32/f64 ret.
            // In that case we return now a tuple with two floats.
            let kind = TyKind::Tup(thin_vec![ty.clone(), ty.clone()]);
            let ty = P(rustc_ast::Ty { kind, id: ty.id, span: ty.span, tokens: None });
            d_decl.output = FnRetTy::Ty(ty);
        }
        if let DiffActivity::DualOnly = x.ret_activity {
            // No need to change the return type,
            // we will just return the shadow in place
            // of the primal return.
        }
    }

    // If we use ActiveOnly, drop the original return value.
    d_decl.output = if active_only_ret { FnRetTy::Default(span) } else { d_decl.output.clone() };

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

    let d_sig = FnSig { header: sig.header.clone(), decl: d_decl, span };
    trace!("Generated signature: {:?}", d_sig);
    (d_sig, new_inputs, idents)
}
