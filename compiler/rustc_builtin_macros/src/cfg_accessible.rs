//! Implementation of the `#[cfg_accessible(path)]` attribute macro.

use rustc_ast as ast;
use rustc_attr_parsing::validate_attr;
use rustc_expand::base::{Annotatable, ExpandResult, ExtCtxt, Indeterminate, MultiItemModifier};
use rustc_feature::AttributeTemplate;
use rustc_span::{Span, sym};

use crate::errors;

pub(crate) struct Expander;

fn validate_input<'a>(ecx: &ExtCtxt<'_>, mi: &'a ast::MetaItem) -> Option<&'a ast::Path> {
    use errors::CfgAccessibleInvalid::*;
    match mi.meta_item_list() {
        None => {}
        Some([]) => {
            ecx.dcx().emit_err(UnspecifiedPath(mi.span));
        }
        Some([_, .., l]) => {
            ecx.dcx().emit_err(MultiplePaths(l.span()));
        }
        Some([nmi]) => match nmi.meta_item() {
            None => {
                ecx.dcx().emit_err(LiteralPath(nmi.span()));
            }
            Some(mi) => {
                if !mi.is_word() {
                    ecx.dcx().emit_err(HasArguments(mi.span));
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
        _is_derive_const: bool,
    ) -> ExpandResult<Vec<Annotatable>, Annotatable> {
        let template = AttributeTemplate { list: Some(&["path"]), ..Default::default() };
        validate_attr::check_builtin_meta_item(
            &ecx.sess.psess,
            meta_item,
            ast::AttrStyle::Outer,
            sym::cfg_accessible,
            template,
            true,
        );

        let Some(path) = validate_input(ecx, meta_item) else {
            return ExpandResult::Ready(Vec::new());
        };

        match ecx.resolver.cfg_accessible(ecx.current_expansion.id, path) {
            Ok(true) => ExpandResult::Ready(vec![item]),
            Ok(false) => ExpandResult::Ready(Vec::new()),
            Err(Indeterminate) if ecx.force_mode => {
                ecx.dcx().emit_err(errors::CfgAccessibleIndeterminate { span });
                ExpandResult::Ready(vec![item])
            }
            Err(Indeterminate) => ExpandResult::Retry(item),
        }
    }
}
