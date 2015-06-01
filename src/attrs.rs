/// checks for attributes

use rustc::plugin::Registry;
use rustc::lint::*;
use syntax::ast::*;
use syntax::ptr::P;
use syntax::codemap::{Span, ExpnInfo};
use syntax::parse::token::InternedString;
use mut_mut::in_macro;

declare_lint! { pub INLINE_ALWAYS, Warn,
    "#[inline(always)] is usually a bad idea."}


#[derive(Copy,Clone)]
pub struct AttrPass;

impl LintPass for AttrPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(INLINE_ALWAYS)
    }
    
    fn check_item(&mut self, cx: &Context, item: &Item) {
		cx.sess().codemap().with_expn_info(item.span.expn_id, 
			|info| check_attrs(cx, info, &item.ident, &item.attrs))
	}
    
    fn check_impl_item(&mut self, cx: &Context, item: &ImplItem) { 
		cx.sess().codemap().with_expn_info(item.span.expn_id, 
			|info| check_attrs(cx, info, &item.ident, &item.attrs))
	}
        
	fn check_trait_item(&mut self, cx: &Context, item: &TraitItem) {
		cx.sess().codemap().with_expn_info(item.span.expn_id, 
			|info| check_attrs(cx, info, &item.ident, &item.attrs))
	}
}

fn check_attrs(cx: &Context, info: Option<&ExpnInfo>, ident: &Ident, 
		attrs: &[Attribute]) {
	if in_macro(cx, info) { return; }
			
	for attr in attrs {
		if let MetaList(ref inline, ref values) = attr.node.value.node {
			if values.len() != 1 || inline != &"inline" { continue; }
			if let MetaWord(ref always) = values[0].node {
				if always != &"always" { continue; }
				cx.span_lint(INLINE_ALWAYS, attr.span, &format!(
					"You have declared #[inline(always)] on {}. This \
					is usually a bad idea. Are you sure?", 
					ident.as_str()));
			}
		}
	}
}
