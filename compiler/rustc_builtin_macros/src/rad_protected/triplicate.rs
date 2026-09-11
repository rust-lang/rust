use rustc_ast as ast;
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::{Span, symbol::Ident, sym, DUMMY_SP};
use thin_vec::thin_vec;
use super::parse_attr_opts::parse_attr_opts;

pub(crate) fn triplicate(
    cx: &mut ExtCtxt<'_>,
    span: Span,
    meta_item: &ast::MetaItem,
    mut item: Annotatable,
) -> Vec<Annotatable> {

    let Some(opts) = parse_attr_opts(cx, meta_item) else {
        return vec![item];
    };

    if opts.triplicate_unsafe() {
        let valid = match &mut item {
            Annotatable::Expr(expr)
                if matches!(&expr.kind, ast::ExprKind::Block(block, _)
                    if matches!(block.rules, ast::BlockCheckMode::Unsafe(_))
                ) => {
                    expr.attrs.push(cx.attr_nested_word(
                        sym::rad_protected_mir,
                        sym::triplicate_unsafe,
                        DUMMY_SP,
                    ));
                    true
                }
            _ => false,
        };

        if !valid {
            cx.dcx().span_err(
                span,
                "`#[rad_protected(triplicate_unsafe)]` can only be applied to `unsafe` blocks",
            );
        }

        return vec![item];
    }

    let Annotatable::Item(mut item) = item else {
        cx.dcx().span_err(span, "`#[rad_protected]` can only be applied to functions");
        return vec![item];
    };

    let ast::Item {
        kind: ast::ItemKind::Fn(func),
        ..
    } = &mut *item
    else {
        cx.dcx().span_err(span, "`#[rad_protected]` can only be applied to functions");
        return vec![Annotatable::Item(item)];
    };

    if func.body.is_none() {
        cx.dcx().span_err(span, "`#[rad_protected]` can only be applied to functions with a body");
        return vec![Annotatable::Item(item)];
    }

    // `shadow_only` drives just the speculative-shadow MIR pass: no process triplication and no
    // checkpoints, so a MIR dump shows that transformation on its own.
    if opts.shadow_only() {
        item.attrs.push(cx.attr_word(sym::rad_protected_shadow, DUMMY_SP));
        return vec![Annotatable::Item(item)];
    }

    let func_body = match &mut item.kind {
        ast::ItemKind::Fn(func) => func.body.as_mut().unwrap(),
        _ => unreachable!("checked above"),
    };

    func_body.stmts.insert(0, cx.stmt_let(
        DUMMY_SP,
        false,
        Ident::new(sym::__guard, DUMMY_SP),
        cx.expr_call_global(
            DUMMY_SP,
            vec![
                Ident::new(sym::std, DUMMY_SP),
                Ident::new(sym::RadRustRuntime, DUMMY_SP),
                Ident::new(sym::triplicate_process, DUMMY_SP),
            ], 
            thin_vec![]
        )
    ));

    let mir_attr = cx.attr_word(sym::rad_protected_mir, DUMMY_SP);
    item.attrs.push(mir_attr);

    vec![Annotatable::Item(item)]
}
