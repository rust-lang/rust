use crate::attr::HasAttrs;
use crate::ast;
use crate::source_map::{ExpnInfo, ExpnFormat};
use crate::ext::base::ExtCtxt;
use crate::ext::build::AstBuilder;
use crate::parse::parser::PathStyle;
use crate::symbol::{Symbol, sym};
use crate::errors::Applicability;

use syntax_pos::Span;

use rustc_data_structures::fx::FxHashSet;

pub fn collect_derives(cx: &mut ExtCtxt<'_>, attrs: &mut Vec<ast::Attribute>) -> Vec<ast::Path> {
    let mut result = Vec::new();
    attrs.retain(|attr| {
        if attr.path != sym::derive {
            return true;
        }
        if !attr.is_meta_item_list() {
            cx.struct_span_err(attr.span, "malformed `derive` attribute input")
                .span_suggestion(
                    attr.span,
                    "missing traits to be derived",
                    "#[derive(Trait1, Trait2, ...)]".to_owned(),
                    Applicability::HasPlaceholders,
                ).emit();
            return false;
        }

        match attr.parse_list(cx.parse_sess,
                              |parser| parser.parse_path_allowing_meta(PathStyle::Mod)) {
            Ok(traits) => {
                result.extend(traits);
                true
            }
            Err(mut e) => {
                e.emit();
                false
            }
        }
    });
    result
}

pub fn add_derived_markers<T>(cx: &mut ExtCtxt<'_>, span: Span, traits: &[ast::Path], item: &mut T)
    where T: HasAttrs,
{
    let (mut names, mut pretty_name) = (FxHashSet::default(), "derive(".to_owned());
    for (i, path) in traits.iter().enumerate() {
        if i > 0 {
            pretty_name.push_str(", ");
        }
        pretty_name.push_str(&path.to_string());
        names.insert(unwrap_or!(path.segments.get(0), continue).ident.name);
    }
    pretty_name.push(')');

    cx.current_expansion.mark.set_expn_info(ExpnInfo::with_unstable(
        ExpnFormat::MacroAttribute(Symbol::intern(&pretty_name)), span, cx.parse_sess.edition,
        &[sym::rustc_attrs, sym::structural_match],
    ));

    let span = span.with_ctxt(cx.backtrace());
    item.visit_attrs(|attrs| {
        if names.contains(&Symbol::intern("Eq")) && names.contains(&Symbol::intern("PartialEq")) {
            let meta = cx.meta_word(span, Symbol::intern("structural_match"));
            attrs.push(cx.attribute(span, meta));
        }
        if names.contains(&Symbol::intern("Copy")) {
            let meta = cx.meta_word(span, sym::rustc_copy_clone_marker);
            attrs.push(cx.attribute(span, meta));
        }
    });
}
