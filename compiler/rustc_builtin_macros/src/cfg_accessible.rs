//! Implementation of the `#[cfg_accessible(path)]` attribute macro.

use rustc_ast as ast;
use rustc_expand::base::{Annotatable, ExpandResult, ExtCtxt, Indeterminate, MultiItemModifier};
use rustc_feature::AttributeTemplate;
use rustc_parse::validate_attr;
use rustc_span::symbol::sym;
use rustc_span::Span;

use crate::errors::{
    MultiplePathsSpecified, NondeterministicAccess, NotSpecified, UnacceptedArguments,
    UnallowedLiteralPath,
};

pub(crate) struct Expander;

fn validate_input<'a>(ecx: &mut ExtCtxt<'_>, mi: &'a ast::MetaItem) -> Option<&'a ast::Path> {
    match mi.meta_item_list() {
        None => {}
        Some([]) => {
            ecx.emit_err(NotSpecified { span: mi.span });
        }
        Some([_, .., l]) => {
            ecx.emit_err(MultiplePathsSpecified { span: l.span() });
        }
        Some([nmi]) => match nmi.meta_item() {
            None => {
                ecx.emit_err(UnallowedLiteralPath { span: nmi.span() });
            }
            Some(mi) => {
                if !mi.is_word() {
                    ecx.emit_err(UnacceptedArguments { span: mi.span });
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

        let Some(path) = validate_input(ecx, meta_item) else {
            return ExpandResult::Ready(Vec::new());
        };

        match ecx.resolver.cfg_accessible(ecx.current_expansion.id, path) {
            Ok(true) => ExpandResult::Ready(vec![item]),
            Ok(false) => ExpandResult::Ready(Vec::new()),
            Err(Indeterminate) if ecx.force_mode => {
                ecx.emit_err(NondeterministicAccess { span });
                ExpandResult::Ready(vec![item])
            }
            Err(Indeterminate) => ExpandResult::Retry(item),
        }
    }
}
