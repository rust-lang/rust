#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
//use crate::util::check_builtin_macro_attribute;
//use crate::util::check_autodiff;

use crate::errors;
use rustc_ast::FnRetTy;
use rustc_ast::expand::autodiff_attrs::{AutoDiffAttrs, DiffActivity, DiffMode};
use rustc_ast::ptr::P;
use rustc_ast::token::{Token, TokenKind};
use rustc_ast::tokenstream::*;
use rustc_ast::{self as ast, FnHeader, FnSig, Generics, MetaItemKind, NestedMetaItem, StmtKind};
use rustc_ast::{BindingAnnotation, ByRef};
use rustc_ast::{Fn, ItemKind, PatKind, Stmt, TyKind, Unsafe};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::symbol::{kw, sym, Ident};
use rustc_span::Span;
use rustc_span::Symbol;
use std::string::String;
use thin_vec::{thin_vec, ThinVec};

fn first_ident(x: &NestedMetaItem) -> rustc_span::symbol::Ident {
    let segments = &x.meta_item().unwrap().path.segments;
    assert!(segments.len() == 1);
    segments[0].ident
}

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
    let (fn_item, has_ret, sig, sig_span) = if let Annotatable::Item(item) = &item
        && let ItemKind::Fn(box ast::Fn { sig, .. }) = &item.kind
    {
        (item, sig.decl.output.has_ret(), sig, ecx.with_call_site_ctxt(sig.span))
    } else {
        ecx.sess.dcx().emit_err(errors::AutoDiffInvalidApplication { span: item.span() });
        return vec![item];
    };
    // create TokenStream from vec elemtents:
    // meta_item doesn't have a .tokens field
    let ts: Vec<Token> = meta_item_vec.clone()[1..]
        .iter()
        .map(|x| {
            let val = first_ident(x);
            let t = Token::from_ast_ident(val);
            t
        })
        .collect();
    let comma: Token = Token::new(TokenKind::Comma, Span::default());
    let mut ts: Vec<TokenTree> = vec![];
    for t in meta_item_vec.clone()[1..].iter() {
        let val = first_ident(t);
        let t = Token::from_ast_ident(val);
        ts.push(TokenTree::Token(t, Spacing::Joint));
        ts.push(TokenTree::Token(comma.clone(), Spacing::Alone));
    }
    let ts: TokenStream = TokenStream::from_iter(ts);

    let x: AutoDiffAttrs = AutoDiffAttrs::from_ast(&meta_item_vec, has_ret);
    dbg!(&x);
    let span = ecx.with_def_site_ctxt(expand_span);

    let (d_sig, old_names, new_args, idents) = gen_enzyme_decl(&sig, &x, span);
    let new_decl_span = d_sig.span;
    let d_body = gen_enzyme_body(
        ecx,
        primal,
        &old_names,
        &new_args,
        span,
        sig_span,
        new_decl_span,
        &sig,
        &d_sig,
        idents,
    );
    let d_ident = meta_item_vec[0].meta_item().unwrap().path.segments[0].ident;

    // The first element of it is the name of the function to be generated
    let asdf = ItemKind::Fn(Box::new(ast::Fn {
        defaultness: ast::Defaultness::Final,
        sig: d_sig,
        generics: Generics::default(),
        body: Some(d_body),
    }));
    let mut rustc_ad_attr =
        P(ast::NormalAttr::from_ident(Ident::with_dummy_span(sym::rustc_autodiff)));
    let mut attr: ast::Attribute = ast::Attribute {
        kind: ast::AttrKind::Normal(rustc_ad_attr.clone()),
        id: ast::AttrId::from_u32(0),
        style: ast::AttrStyle::Outer,
        span: span,
    };
    orig_item.attrs.push(attr.clone());

    // Now update for d_fn
    rustc_ad_attr.item.args = rustc_ast::AttrArgs::Delimited(rustc_ast::DelimArgs {
        dspan: DelimSpan::dummy(),
        delim: rustc_ast::token::Delimiter::Parenthesis,
        tokens: ts,
    });
    let mut attr2: ast::Attribute = ast::Attribute {
        kind: ast::AttrKind::Normal(rustc_ad_attr),
        id: ast::AttrId::from_u32(0),
        style: ast::AttrStyle::Outer,
        span: span,
    };
    let attr_vec: rustc_ast::AttrVec = thin_vec![attr2];
    let d_fn = ecx.item(span, d_ident, attr_vec, asdf);

    let orig_annotatable = Annotatable::Item(orig_item.clone());
    let d_annotatable = Annotatable::Item(d_fn);
    return vec![orig_annotatable, d_annotatable];
}

// shadow arguments must be mutable references or ptrs, because Enzyme will write into them.
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
fn gen_enzyme_body(
    ecx: &ExtCtxt<'_>,
    primal: Ident,
    old_names: &[String],
    new_names: &[String],
    span: Span,
    sig_span: Span,
    new_decl_span: Span,
    sig: &ast::FnSig,
    d_sig: &ast::FnSig,
    idents: Vec<Ident>,
) -> P<ast::Block> {
    let blackbox_path = ecx.std_path(&[Symbol::intern("hint"), Symbol::intern("black_box")]);
    let zeroed_path = ecx.std_path(&[Symbol::intern("mem"), Symbol::intern("zeroed")]);
    let empty_loop_block = ecx.block(span, ThinVec::new());
    let loop_expr = ecx.expr_loop(span, empty_loop_block);

    let blackbox_call_expr = ecx.expr_path(ecx.path(span, blackbox_path));
    let zeroed_call_expr = ecx.expr_path(ecx.path(span, zeroed_path));

    let mem_zeroed_call: Stmt =
        ecx.stmt_expr(ecx.expr_call(span, zeroed_call_expr.clone(), thin_vec![]));
    let unsafe_block_with_zeroed_call: P<ast::Expr> = ecx.expr_block(P(ast::Block {
        stmts: thin_vec![mem_zeroed_call],
        id: ast::DUMMY_NODE_ID,
        rules: ast::BlockCheckMode::Unsafe(ast::UserProvided),
        span: sig_span,
        tokens: None,
        could_be_bare_literal: false,
    }));
    let primal_call = gen_primal_call(ecx, span, primal, sig, idents);
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
    body.stmts.push(ecx.stmt_semi(primal_call));
    body.stmts.push(ecx.stmt_semi(black_box_primal_call));
    body.stmts.push(ecx.stmt_semi(black_box_remaining_args));
    body.stmts.push(ecx.stmt_expr(loop_expr));
    body
}

fn gen_primal_call(
    ecx: &ExtCtxt<'_>,
    span: Span,
    primal: Ident,
    sig: &ast::FnSig,
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
fn gen_enzyme_decl(
    sig: &ast::FnSig,
    x: &AutoDiffAttrs,
    span: Span,
) -> (ast::FnSig, Vec<String>, Vec<String>, Vec<Ident>) {
    assert!(sig.decl.inputs.len() == x.input_activity.len());
    assert!(sig.decl.output.has_ret() == x.has_ret_activity());
    let mut d_decl = sig.decl.clone();
    let mut d_inputs = Vec::new();
    let mut new_inputs = Vec::new();
    let mut old_names = Vec::new();
    let mut idents = Vec::new();
    let mut act_ret = ThinVec::new();
    for (arg, activity) in sig.decl.inputs.iter().zip(x.input_activity.iter()) {
        d_inputs.push(arg.clone());
        match activity {
            DiffActivity::Active => {
                assert!(x.mode == DiffMode::Reverse);
                act_ret.push(arg.ty.clone());
            }
            DiffActivity::Duplicated | DiffActivity::Dual => {
                let mut shadow_arg = arg.clone();
                shadow_arg.ty = P(assure_mut_ref(&arg.ty));
                // adjust name depending on mode
                let old_name = if let PatKind::Ident(_, ident, _) = arg.pat.kind {
                    ident.name
                } else {
                    dbg!(&shadow_arg.pat);
                    panic!("not an ident?");
                };
                old_names.push(old_name.to_string());
                let name: String = match x.mode {
                    DiffMode::Reverse => format!("d{}", old_name),
                    DiffMode::Forward => format!("b{}", old_name),
                    _ => panic!("unsupported mode: {}", old_name),
                };
                dbg!(&name);
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
            _ => {
                dbg!(&activity);
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
        let mut ret_ty = match d_decl.output {
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
    (d_sig, old_names, new_inputs, idents)
}
