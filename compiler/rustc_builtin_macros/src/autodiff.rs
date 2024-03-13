#![allow(unused_imports)]
//use crate::util::check_builtin_macro_attribute;
//use crate::util::check_autodiff;

use crate::errors;
use rustc_ast::expand::autodiff_attrs::{AutoDiffAttrs, DiffActivity, DiffMode, valid_input_activity, valid_ty_for_activity};
use rustc_ast::ptr::P;
use rustc_ast::token::{Token, TokenKind};
use rustc_ast::tokenstream::*;
use rustc_ast::FnRetTy;
use rustc_ast::{self as ast, FnHeader, FnSig, Generics, MetaItemKind, NestedMetaItem, StmtKind};
use rustc_ast::{BindingAnnotation, ByRef};
use rustc_ast::{Fn, ItemKind, PatKind, Stmt, TyKind, Unsafe};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::symbol::{kw, sym, Ident};
use rustc_span::Span;
use rustc_span::Symbol;
use std::string::String;
use thin_vec::{thin_vec, ThinVec};
use std::str::FromStr;

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
pub fn from_ast(ecx: &mut ExtCtxt<'_>, meta_item: &ThinVec<NestedMetaItem>, has_ret: bool) -> AutoDiffAttrs {

    let mode = name(&meta_item[1]);
    let mode = match DiffMode::from_str(&mode) {
        Ok(x) => x,
        Err(_) => {
            ecx.sess.dcx().emit_err(errors::AutoDiffInvalidMode { span: meta_item[1].span(), mode});
            return AutoDiffAttrs::inactive();
        },
    };
    let activities: Vec<DiffActivity> = meta_item[2..]
        .iter()
        .map(|x| {
            let activity_str = name(&x);
            DiffActivity::from_str(&activity_str).unwrap()
        })
        .collect();

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
    item: Annotatable,
) -> Vec<Annotatable> {
    //check_builtin_macro_attribute(ecx, meta_item, sym::alloc_error_handler);

    let meta_item_vec: ThinVec<NestedMetaItem> = match meta_item.kind {
        ast::MetaItemKind::List(ref vec) => vec.clone(),
        _ => {
            ecx.sess.dcx().emit_err(errors::AutoDiffInvalidApplication { span: item.span() });
            return vec![item];
        }
    };
    let mut orig_item: P<ast::Item> = item.clone().expect_item();
    let primal = orig_item.ident.clone();

    // Allow using `#[autodiff(...)]` only on a Fn
    let (has_ret, sig, sig_span) = if let Annotatable::Item(item) = &item
        && let ItemKind::Fn(box ast::Fn { sig, .. }) = &item.kind
    {
        (sig.decl.output.has_ret(), sig, ecx.with_call_site_ctxt(sig.span))
    } else {
        ecx.sess.dcx().emit_err(errors::AutoDiffInvalidApplication { span: item.span() });
        return vec![item];
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
    let ts: TokenStream = TokenStream::from_iter(ts);

    let x: AutoDiffAttrs = from_ast(ecx, &meta_item_vec, has_ret);
    if !x.is_active() {
        // We encountered an error, so we return the original item.
        // This allows us to potentially parse other attributes.
        return vec![item];
    }
    dbg!(&x);
    let span = ecx.with_def_site_ctxt(expand_span);

    let n_active: u32 = x.input_activity.iter()
        .filter(|a| **a == DiffActivity::Active || **a == DiffActivity::ActiveOnly)
        .count() as u32;
    let (d_sig, new_args, idents) = gen_enzyme_decl(ecx, &sig, &x, span);
    let new_decl_span = d_sig.span;
    let d_body = gen_enzyme_body(
        ecx,
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
    let asdf = ItemKind::Fn(Box::new(ast::Fn {
        defaultness: ast::Defaultness::Final,
        sig: d_sig,
        generics: Generics::default(),
        body: Some(d_body),
    }));
    let mut rustc_ad_attr =
        P(ast::NormalAttr::from_ident(Ident::with_dummy_span(sym::rustc_autodiff)));
    let ts2: Vec<TokenTree> = vec![
            TokenTree::Token(
            Token::new(TokenKind::Ident(sym::never, false), span),
            Spacing::Joint,
        )];
    let never_arg = ast::DelimArgs {
        dspan: ast::tokenstream::DelimSpan::from_single(span),
        delim: ast::token::Delimiter::Parenthesis,
        tokens: ast::tokenstream::TokenStream::from_iter(ts2),
    };
    let inline_item = ast::AttrItem {
        path: ast::Path::from_ident(Ident::with_dummy_span(sym::inline)),
        args: ast::AttrArgs::Delimited(never_arg),
        tokens: None,
    };
    let inline_never_attr = P(ast::NormalAttr {
        item: inline_item,
        tokens: None,
    });
    let mut attr: ast::Attribute = ast::Attribute {
        kind: ast::AttrKind::Normal(rustc_ad_attr.clone()),
        //id: ast::DUMMY_TR_ID,
        id: ast::AttrId::from_u32(12341), // TODO: fix
        style: ast::AttrStyle::Outer,
        span,
    };
    let inline_never : ast::Attribute = ast::Attribute {
        kind: ast::AttrKind::Normal(inline_never_attr),
        //id: ast::DUMMY_TR_ID,
        id: ast::AttrId::from_u32(12342), // TODO: fix
        style: ast::AttrStyle::Outer,
        span,
    };
    // don't add it multiple times:
    if !orig_item.attrs.iter().any(|a| a.id == attr.id) {
        orig_item.attrs.push(attr.clone());
    }
    if !orig_item.attrs.iter().any(|a| a.id == inline_never.id) {
        orig_item.attrs.push(inline_never);
    }

    // Now update for d_fn
    rustc_ad_attr.item.args = rustc_ast::AttrArgs::Delimited(rustc_ast::DelimArgs {
        dspan: DelimSpan::dummy(),
        delim: rustc_ast::token::Delimiter::Parenthesis,
        tokens: ts,
    });
    attr.kind = ast::AttrKind::Normal(rustc_ad_attr);
    let d_fn = ecx.item(span, d_ident, thin_vec![attr], asdf);

    let orig_annotatable = Annotatable::Item(orig_item);
    let d_annotatable = Annotatable::Item(d_fn);
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
    //let default_path = ecx.def_site_path(&[Symbol::intern("f32"), Symbol::intern("default")]);
    let empty_loop_block = ecx.block(span, ThinVec::new());
    let noop = ast::InlineAsm {
        template: vec![ast::InlineAsmTemplatePiece::String("NOP".to_string())],
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
    //let default_call_expr = ecx.expr_path(ecx.path(span, default_path));
    let primal_call = gen_primal_call(ecx, span, primal, idents);
    // create ::core::hint::black_box(array(arr));
    let black_box_primal_call =
        ecx.expr_call(new_decl_span, blackbox_call_expr.clone(), thin_vec![primal_call.clone()]);
    // create ::core::hint::black_box((grad_arr, tang_y));
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

    let primal_ret = sig.decl.output.has_ret();

    if primal_ret && n_active == 0 {
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

    let mut exprs = ThinVec::<P::<ast::Expr>>::new();
    if primal_ret {
        // We have both primal ret and active floats.
        // primal ret is first, by construction.
        exprs.push(primal_call.clone());
    }

    // Now construct default placeholder for each active float.
    // Is there something nicer than f32::default() and f64::default()?
    let mut d_ret_ty = match d_sig.decl.output {
        FnRetTy::Ty(ref ty) => ty.clone(),
        FnRetTy::Default(span) => {
            panic!("Did not expect Default ret ty: {:?}", span);
        }
    };
    let mut d_ret_ty = match d_ret_ty.kind {
        TyKind::Tup(ref mut tys) => {
            tys.clone()
        }
        _ => {
            // We messed up construction of d_sig
            panic!("Did not expect non-tuple ret ty: {:?}", d_ret_ty);
        }
    };
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
    };

    let ret_tuple: P<ast::Expr> = ecx.expr_tuple(span, exprs);
    let ret = ecx.expr_call(new_decl_span, blackbox_call_expr.clone(), thin_vec![ret_tuple]);
    body.stmts.push(ecx.stmt_expr(ret));
    //body.stmts.push(ecx.stmt_expr(ret_tuple));

    body
}

#[cfg(llvm_enzyme)]
fn gen_primal_call(
    ecx: &ExtCtxt<'_>,
    span: Span,
    primal: Ident,
    idents: Vec<Ident>,
) -> P<ast::Expr> {
    let primal_call_expr = ecx.expr_path(ecx.path_ident(span, primal));
    let args = idents.iter().map(|arg| ecx.expr_path(ecx.path_ident(span, *arg))).collect();
    let primal_call = ecx.expr_call(span, primal_call_expr, args);
    primal_call
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
    assert!(sig.decl.inputs.len() == x.input_activity.len());
    assert!(sig.decl.output.has_ret() == x.has_ret_activity());
    let mut d_decl = sig.decl.clone();
    let mut d_inputs = Vec::new();
    let mut new_inputs = Vec::new();
    //let mut old_names = Vec::new();
    let mut idents = Vec::new();
    let mut act_ret = ThinVec::new();
    for (arg, activity) in sig.decl.inputs.iter().zip(x.input_activity.iter()) {
        d_inputs.push(arg.clone());
        if !valid_input_activity(x.mode, *activity) {
            ecx.sess.dcx().emit_err(errors::AutoDiffInvalidApplicationModeAct {
                span,
                mode: x.mode.to_string(),
                act: activity.to_string()
            });
        }
        if !valid_ty_for_activity(&arg.ty, *activity) {
            ecx.sess.dcx().emit_err(errors::AutoDiffInvalidTypeForActivity {
                span: arg.ty.span,
                act: activity.to_string()
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
                    kind: PatKind::Ident(BindingAnnotation::NONE, ident, None),
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
                    kind: PatKind::Ident(BindingAnnotation::NONE, ident, None),
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

    // If we return a scalar in the primal and the scalar is active,
    // then add it as last arg to the inputs.
    if let DiffMode::Reverse = x.mode {
        if let DiffActivity::Active = x.ret_activity {
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
                    kind: PatKind::Ident(BindingAnnotation::NONE, ident, None),
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
    }
    d_decl.inputs = d_inputs.into();

    // If we have an active input scalar, add it's gradient to the
    // return type. This might require changing the return type to a
    // tuple.
    if act_ret.len() > 0 {
        let ret_ty = match d_decl.output {
            FnRetTy::Ty(ref ty) => {
                act_ret.insert(0, ty.clone());
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
    (d_sig, new_inputs, idents)
}
