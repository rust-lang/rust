//! "splat" expansion transforms a function with a tuple argument, into a function taking separate
//! arguments. It's like `unboxed_closures`, but more general.

use rustc_ast as ast;
use rustc_expand::base::*;
use rustc_span::{Span, sym};

use crate::errors;
use crate::util::{check_builtin_macro_attribute, warn_on_duplicate_attribute};

/// Expand a function taking a tuple, into a function taking separate arguments.
/// FIXME(splat): not yet implemented, currently passes the input through unchanged, after checking it's a function.
pub(crate) fn expand_splat(
    ecx: &mut ExtCtxt<'_>,
    _attr_sp: Span,
    meta_item: &ast::MetaItem,
    anno_item: Annotatable,
) -> Vec<Annotatable> {
    check_builtin_macro_attribute(ecx, meta_item, sym::splat);
    warn_on_duplicate_attribute(ecx, &anno_item, sym::splat);

    let span = anno_item.span();
    let (item_kind, item, assoc_item, foreign_item) = match anno_item {
        Annotatable::Item(item) => (item.kind.clone(), Some((item, false)), None, None),
        Annotatable::Stmt(stmt) if let ast::StmtKind::Item(_) = stmt.kind => {
            if let ast::StmtKind::Item(item) = stmt.kind {
                (item.kind.clone(), Some((item, true)), None, None)
            } else {
                unreachable!()
            }
        }
        Annotatable::AssocItem(item, assoc_ctx) => {
            (ast::ItemKind::from(item.kind.clone()), None, Some((item, assoc_ctx)), None)
        }
        Annotatable::ForeignItem(item) => {
            (ast::ItemKind::from(item.kind.clone()), None, None, Some(item))
        }
        _ => {
            ecx.dcx().emit_err(errors::SplatNonItem { span });
            return vec![];
        }
    };

    // `#[splat]` is valid on functions, associated trait methods, impl methods, and extern
    // functions. Only modify the item in those cases.
    // FIXME(splat): combine this match with the match above, to avoid cloning ItemKind
    match &item_kind {
        ast::ItemKind::Fn(box ast::Fn { .. })
        | ast::ItemKind::Trait(box ast::Trait { .. })
        | ast::ItemKind::Impl(ast::Impl { .. })
        | ast::ItemKind::ForeignMod(ast::ForeignMod { .. }) => {
            // FIXME(splat): do the splat here
        }
        _ => {
            ecx.dcx().emit_err(errors::SplatNonFunction { span });
            return vec![];
        }
    }

    let ret = if let Some((item, is_stmt)) = item {
        if is_stmt {
            Annotatable::Stmt(Box::new(ecx.stmt_item(item.span, item)))
        } else {
            Annotatable::Item(item)
        }
    } else if let Some((item, assoc_ctx)) = assoc_item {
        Annotatable::AssocItem(item, assoc_ctx)
    } else if let Some(item) = foreign_item {
        Annotatable::ForeignItem(item)
    } else {
        unreachable!();
    };

    vec![ret]
}
