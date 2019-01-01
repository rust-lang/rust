use attr::HasAttrs;
use ast;
use source_map::{hygiene, ExpnInfo, ExpnFormat};
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use parse::parser::PathStyle;
use symbol::Symbol;
use syntax_pos::Span;

use rustc_data_structures::fx::FxHashSet;

pub fn collect_derives(cx: &mut ExtCtxt, attrs: &mut Vec<ast::Attribute>) -> Vec<ast::Path> {
    let mut result = Vec::new();
    attrs.retain(|attr| {
        if attr.path != "derive" {
            return true;
        }
        if !attr.is_meta_item_list() {
            cx.span_err(attr.span,
                        "attribute must be of the form `#[derive(Trait1, Trait2, ...)]`");
            return false;
        }

        match attr.parse_list(cx.parse_sess,
                              |parser| parser.parse_path_allowing_meta(PathStyle::Mod)) {
            Ok(ref traits) if traits.is_empty() => {
                cx.span_warn(attr.span, "empty trait list in `derive`");
                false
            }
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

pub fn add_derived_markers<T>(cx: &mut ExtCtxt, span: Span, traits: &[ast::Path], item: T) -> T
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

    cx.current_expansion.mark.set_expn_info(ExpnInfo {
        call_site: span,
        def_site: None,
        format: ExpnFormat::MacroAttribute(Symbol::intern(&pretty_name)),
        allow_internal_unstable: true,
        allow_internal_unsafe: false,
        local_inner_macros: false,
        edition: hygiene::default_edition(),
    });

    let span = span.with_ctxt(cx.backtrace());
    item.map_attrs(|mut attrs| {
        if names.contains(&Symbol::intern("Eq")) && names.contains(&Symbol::intern("PartialEq")) {
            let meta = cx.meta_word(span, Symbol::intern("structural_match"));
            attrs.push(cx.attribute(span, meta));
        }
        if names.contains(&Symbol::intern("Copy")) {
            let meta = cx.meta_word(span, Symbol::intern("rustc_copy_clone_marker"));
            attrs.push(cx.attribute(span, meta));
        }
        attrs
    })
}
