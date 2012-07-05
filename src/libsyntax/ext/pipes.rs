
import codemap::span;
import ext::base::ext_ctxt;

fn expand_proto(cx: ext_ctxt, span: span, id: ast::ident, tt: ast::token_tree)
    -> @ast::item
{
    cx.span_unimpl(span,
                   "Protocol compiler")
}