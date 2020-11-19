//! Implementation of the `#[cfg_accessible(path)]` attribute macro.

use rustc_ast as ast;
use rustc_expand::base::{Annotatable, ExpandResult, ExtCtxt, Indeterminate, MultiItemModifier};
use rustc_feature::AttributeTemplate;
use rustc_parse::validate_attr;
use rustc_span::symbol::sym;
use rustc_span::Span;

crate struct Expander;

fn validate_input<'a>(ecx: &mut ExtCtxt<'_>, mi: &'a ast::MetaItem) -> Option<&'a ast::Path> {
    match mi.meta_item_list() {
        None => {}
        Some([]) => ecx.span_err(mi.span, "`cfg_accessible` path is not specified"),
        Some([_, .., l]) => ecx.span_err(l.span(), "multiple `cfg_accessible` paths are specified"),
        Some([nmi]) => match nmi.meta_item() {
            None => ecx.span_err(nmi.span(), "`cfg_accessible` path cannot be a literal"),
            Some(mi) => {
                if !mi.is_word() {
                    ecx.span_err(mi.span, "`cfg_accessible` path cannot accept arguments");
                }
                return Some(&mi.path);
            }
        },
    }
    None
}

impl MultiItemModifier for Expander {
    fn expand(
        &self,
        ecx: &mut ExtCtxt<'_>,
        span: Span,
        meta_item: &ast::MetaItem,
        item: Annotatable,
    ) -> ExpandResult<Vec<Annotatable>, Annotatable> {
        let template = AttributeTemplate { list: Some("path"), ..Default::default() };
        let attr = &ecx.attribute(meta_item.clone());
        validate_attr::check_builtin_attribute(
            &ecx.sess.parse_sess,
            attr,
            sym::cfg_accessible,
            template,
        );

        let path = match validate_input(ecx, meta_item) {
            Some(path) => path,
            None => return ExpandResult::Ready(Vec::new()),
        };

        match ecx.resolver.cfg_accessible(ecx.current_expansion.id, path) {
            Ok(true) => ExpandResult::Ready(vec![item]),
            Ok(false) => ExpandResult::Ready(Vec::new()),
            Err(Indeterminate) if ecx.force_mode => {
                ecx.span_err(span, "cannot determine whether the path is accessible or not");
                ExpandResult::Ready(vec![item])
            }
            Err(Indeterminate) => ExpandResult::Retry(item),
        }
    }
}
