use rustc_ast::{self as ast, AttrArgs, AttrItem, AttrStyle, Safety, StmtKind, attr};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::{Span, sym};

use crate::diagnostics;
use crate::util::check_builtin_macro_attribute;

pub(crate) fn expand(
    ecx: &mut ExtCtxt<'_>,
    span: Span,
    meta_item: &ast::MetaItem,
    mut item: Annotatable,
) -> Vec<Annotatable> {
    check_builtin_macro_attribute(ecx, meta_item, sym::global_allocator);

    // Allow using `#[panic_handler]` on an item statement
    // FIXME - if we get deref patterns, use them to reduce duplication here
    let attrs = if let Annotatable::Item(item) = &mut item {
        &mut item.attrs
    } else if let Annotatable::Stmt(stmt) = &mut item
        && let StmtKind::Item(item) = &mut stmt.kind
    {
        &mut item.attrs
    } else {
        ecx.dcx().emit_err(diagnostics::PanicHandlerMustBeFn { span: item.span() });
        return vec![item];
    };

    attrs.push(attr::mk_attr_from_item(
        &ecx.sess.psess.attr_id_generator,
        AttrItem {
            unsafety: Safety::Unsafe(span),
            path: ecx.path_global(span, ecx.std_path(&[sym::panicking, sym::panic_handler])),
            args: AttrArgs::Empty,
            span,
        },
        None,
        AttrStyle::Outer,
        span,
    ));

    vec![item]
}
