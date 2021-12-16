use rustc_ast::{Attribute, MetaItem};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_feature::AttributeTemplate;
use rustc_lint_defs::builtin::DUPLICATE_MACRO_ATTRIBUTES;
use rustc_parse::validate_attr;
use rustc_span::Symbol;

pub fn check_builtin_macro_attribute(ecx: &ExtCtxt<'_>, meta_item: &MetaItem, name: Symbol) {
    // All the built-in macro attributes are "words" at the moment.
    let template = AttributeTemplate { word: true, ..Default::default() };
    let attr = ecx.attribute(meta_item.clone());
    validate_attr::check_builtin_attribute(&ecx.sess.parse_sess, &attr, name, template);
}

/// Emit a warning if the item is annotated with the given attribute. This is used to diagnose when
/// an attribute may have been mistakenly duplicated.
pub fn warn_on_duplicate_attribute(ecx: &ExtCtxt<'_>, item: &Annotatable, name: Symbol) {
    let attrs: Option<&[Attribute]> = match item {
        Annotatable::Item(item) => Some(&item.attrs),
        Annotatable::TraitItem(item) => Some(&item.attrs),
        Annotatable::ImplItem(item) => Some(&item.attrs),
        Annotatable::ForeignItem(item) => Some(&item.attrs),
        Annotatable::Expr(expr) => Some(&expr.attrs),
        Annotatable::Arm(arm) => Some(&arm.attrs),
        Annotatable::ExprField(field) => Some(&field.attrs),
        Annotatable::PatField(field) => Some(&field.attrs),
        Annotatable::GenericParam(param) => Some(&param.attrs),
        Annotatable::Param(param) => Some(&param.attrs),
        Annotatable::FieldDef(def) => Some(&def.attrs),
        Annotatable::Variant(variant) => Some(&variant.attrs),
        _ => None,
    };
    if let Some(attrs) = attrs {
        if let Some(attr) = ecx.sess.find_by_name(attrs, name) {
            ecx.parse_sess().buffer_lint(
                DUPLICATE_MACRO_ATTRIBUTES,
                attr.span,
                ecx.current_expansion.lint_node_id,
                "duplicated attribute",
            );
        }
    }
}
