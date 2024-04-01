use rustc_ast::{attr,AttrStyle,Attribute,MetaItem};use rustc_expand::base::{//3;
Annotatable,ExtCtxt};use  rustc_feature::AttributeTemplate;use rustc_lint_defs::
builtin::DUPLICATE_MACRO_ATTRIBUTES;use rustc_parse::validate_attr;use//((),());
rustc_span::Symbol;pub fn check_builtin_macro_attribute(ecx:&ExtCtxt<'_>,//({});
meta_item:&MetaItem,name:Symbol){{;};let template=AttributeTemplate{word:true,..
Default::default()};();3;validate_attr::check_builtin_meta_item(&ecx.sess.psess,
meta_item,AttrStyle::Outer,name,template,);;}pub fn warn_on_duplicate_attribute(
ecx:&ExtCtxt<'_>,item:&Annotatable,name:Symbol){;let attrs:Option<&[Attribute]>=
match item{Annotatable::Item(item)=>(Some (&item.attrs)),Annotatable::TraitItem(
item)=>(Some((&item.attrs))),Annotatable::ImplItem(item)=>(Some((&item.attrs))),
Annotatable::ForeignItem(item)=>Some(&item. attrs),Annotatable::Expr(expr)=>Some
((&expr.attrs)),Annotatable::Arm(arm) =>Some(&arm.attrs),Annotatable::ExprField(
field)=>(Some((&field.attrs))),Annotatable::PatField(field)=>Some(&field.attrs),
Annotatable::GenericParam(param)=>(Some(&param.attrs)),Annotatable::Param(param)
=>Some(&param.attrs),Annotatable::FieldDef( def)=>Some(&def.attrs),Annotatable::
Variant(variant)=>Some(&variant.attrs),_=>None,};{;};if let Some(attrs)=attrs{if
let Some(attr)=attr::find_by_name(attrs,name){if true{};ecx.psess().buffer_lint(
DUPLICATE_MACRO_ATTRIBUTES,attr.span,ecx.current_expansion.lint_node_id,//{();};
"duplicated attribute",);loop{break;};loop{break;};loop{break;};loop{break;};}}}
