#![allow(unused_imports)]
//use crate::util::check_builtin_macro_attribute;
//use crate::util::check_autodiff;

use std::string::String;
use crate::errors;
use rustc_ast::ptr::P;
use rustc_ast::{BindingAnnotation, ByRef};
use rustc_ast::{self as ast, FnHeader, FnSig, Generics, StmtKind, NestedMetaItem, MetaItemKind};
use rustc_ast::{Fn, ItemKind, Stmt, TyKind, Unsafe, PatKind};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::symbol::{kw, sym, Ident};
use rustc_span::Span;
use thin_vec::{thin_vec, ThinVec};
use rustc_span::Symbol;
use rustc_ast::expand::autodiff_attrs::{AutoDiffAttrs, DiffActivity, DiffMode};

pub fn expand(
    ecx: &mut ExtCtxt<'_>,
    _span: Span,
    meta_item: &ast::MetaItem,
    item: Annotatable,
) -> Vec<Annotatable> {
    //check_builtin_macro_attribute(ecx, meta_item, sym::alloc_error_handler);

    let meta_item_vec: ThinVec<NestedMetaItem> = match meta_item.kind {
        ast::MetaItemKind::List(ref vec) => vec.clone(),
        _ => {
            ecx.sess
                .dcx()
                .emit_err(errors::AutoDiffInvalidApplication { span: item.span() });
            return vec![item];
        }
    };
    let input = item.clone();
    let orig_item: P<ast::Item> = item.clone().expect_item();
    let mut d_item: P<ast::Item> = item.clone().expect_item();
    let primal = orig_item.ident.clone();

    // Allow using `#[autodiff(...)]` only on a Fn
    let (fn_item, has_ret, sig, sig_span) = if let Annotatable::Item(item) = &item
        && let ItemKind::Fn(box ast::Fn { sig, .. }) = &item.kind
    {
        (item, sig.decl.output.has_ret(), sig, ecx.with_def_site_ctxt(sig.span))
    } else {
        ecx.sess
            .dcx()
            .emit_err(errors::AutoDiffInvalidApplication { span: item.span() });
        return vec![input];
    };
    let x: AutoDiffAttrs = AutoDiffAttrs::from_ast(&meta_item_vec, has_ret);
    dbg!(&x);
    let span = ecx.with_def_site_ctxt(fn_item.span);

    let (d_decl, old_names, new_args) = gen_enzyme_decl(ecx, &sig.decl, &x, span, sig_span);
    let d_body = gen_enzyme_body(ecx, primal, &old_names, &new_args, span, sig_span);
    let meta_item_name = meta_item_vec[0].meta_item().unwrap();
    d_item.ident = meta_item_name.path.segments[0].ident;
    // update d_item
    if let ItemKind::Fn(box ast::Fn { sig, body, .. }) = &mut d_item.kind {
        *sig.decl = d_decl;
        *body = Some(d_body);
    } else {
        ecx.sess
            .dcx()
            .emit_err(errors::AutoDiffInvalidApplication { span: item.span() });
        return vec![input];
    }

    let orig_annotatable = Annotatable::Item(orig_item.clone());
    let d_annotatable = Annotatable::Item(d_item.clone());
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


// The body of our generated functions will consist of three black_Box calls.
// The first will call the primal function with the original arguments.
// The second will just take the shadow arguments.
// The third will (unsafely) call std::mem::zeroed(), to match the return type of the new function
// (whatever that might be). This way we surpress rustc from optimizing anyt argument away.
fn gen_enzyme_body(ecx: &ExtCtxt<'_>, primal: Ident, old_names: &[String], new_names: &[String], span: Span, sig_span: Span) -> P<ast::Block> {
    let blackbox_path = ecx.std_path(&[Symbol::intern("hint"), Symbol::intern("black_box")]);
    let zeroed_path = ecx.std_path(&[Symbol::intern("mem"), Symbol::intern("zeroed")]);

    let primal_call_expr = ecx.expr_path(ecx.path_ident(span, primal));
    let blackbox_call_expr = ecx.expr_path(ecx.path(span, blackbox_path));
    let zeroed_call_expr = ecx.expr_path(ecx.path(span, zeroed_path));

    let mem_zeroed_call: Stmt = ecx.stmt_expr(ecx.expr_call(
        span,
        zeroed_call_expr.clone(),
        thin_vec![],
    ));
    let unsafe_block_with_zeroed_call: P<ast::Expr> = ecx.expr_block(P(ast::Block {
        stmts: thin_vec![mem_zeroed_call],
        id: ast::DUMMY_NODE_ID,
        rules: ast::BlockCheckMode::Unsafe(ast::UserProvided),
        span: sig_span,
        tokens: None,
        could_be_bare_literal: false,
    }));
    // create ::core::hint::black_box(array(arr));
    let _primal_call = ecx.expr_call(
        span,
        primal_call_expr.clone(),
        old_names.iter().map(|name| {
            ecx.expr_path(ecx.path_ident(span, Ident::from_str(name)))
        }).collect(),
    );

    // create ::core::hint::black_box(grad_arr, tang_y));
    let black_box1 = ecx.expr_call(
        sig_span,
        blackbox_call_expr.clone(),
        new_names.iter().map(|arg| {
            ecx.expr_path(ecx.path_ident(span, Ident::from_str(arg)))
        }).collect(),
    );

    // create ::core::hint::black_box(unsafe { ::core::mem::zeroed() })
    let black_box2 = ecx.expr_call(
        sig_span,
        blackbox_call_expr.clone(),
        thin_vec![unsafe_block_with_zeroed_call.clone()],
    );

    let mut body = ecx.block(span, ThinVec::new());
    body.stmts.push(ecx.stmt_expr(black_box1));
    body.stmts.push(ecx.stmt_expr(black_box2));
    body
}

// Generate the new function declaration. Const arguments are kept as is. Duplicated arguments must
// be pointers or references. Those receive a shadow argument, which is a mutable reference/pointer.
// Active arguments must be scalars. Their shadow argument is added to the return type (and will be
// zero-initialized by Enzyme). Active arguments are not handled yet.
// Each argument of the primal function (and the return type if existing) must be annotated with an
// activity.
fn gen_enzyme_decl(_ecx: &ExtCtxt<'_>, decl: &ast::FnDecl, x: &AutoDiffAttrs, _span: Span, _sig_span: Span)
        -> (ast::FnDecl, Vec<String>, Vec<String>) {
    assert!(decl.inputs.len() == x.input_activity.len());
    assert!(decl.output.has_ret() == x.has_ret_activity());
    let mut d_decl = decl.clone();
    let mut d_inputs = Vec::new();
    let mut new_inputs = Vec::new();
    let mut old_names = Vec::new();
    for (arg, activity) in decl.inputs.iter().zip(x.input_activity.iter()) {
        dbg!(&arg);
        d_inputs.push(arg.clone());
        match activity {
            DiffActivity::Duplicated => {
                let mut shadow_arg = arg.clone();
                shadow_arg.ty = P(assure_mut_ref(&arg.ty));
                // adjust name depending on mode
                let old_name = if let PatKind::Ident(_, ident, _) = shadow_arg.pat.kind {
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
                shadow_arg.pat = P(ast::Pat {
                    // TODO: Check id
                    id: ast::DUMMY_NODE_ID,
                    kind: PatKind::Ident(BindingAnnotation::NONE,
                        Ident::from_str_and_span(&name, shadow_arg.pat.span),
                        None,
                    ),
                    span: shadow_arg.pat.span,
                    tokens: shadow_arg.pat.tokens.clone(),
                });

                d_inputs.push(shadow_arg);
            }
            _ => {},
        }
    }
    d_decl.inputs = d_inputs.into();
    (d_decl, old_names, new_inputs)
}
