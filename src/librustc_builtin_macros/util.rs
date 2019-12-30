use rustc_ast::ast::MetaItem;
use rustc_expand::base::ExtCtxt;
use rustc_feature::AttributeTemplate;
use rustc_parse::validate_attr;
use rustc_span::Symbol;

pub fn check_builtin_macro_attribute(ecx: &ExtCtxt<'_>, meta_item: &MetaItem, name: Symbol) {
    // All the built-in macro attributes are "words" at the moment.
    let template = AttributeTemplate { word: true, ..Default::default() };
    let attr = ecx.attribute(meta_item.clone());
    validate_attr::check_builtin_attribute(ecx.parse_sess, &attr, name, template);
}
