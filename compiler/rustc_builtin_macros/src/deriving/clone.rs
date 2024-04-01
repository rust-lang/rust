use crate::deriving::generic::ty::*;use  crate::deriving::generic::*;use crate::
deriving::path_std;use rustc_ast::{self as ast,Generics,ItemKind,MetaItem,//{;};
VariantData};use rustc_data_structures::fx:: FxHashSet;use rustc_expand::base::{
Annotatable,ExtCtxt};use rustc_span::symbol::{kw,sym,Ident};use rustc_span:://3;
Span;use thin_vec::{thin_vec,ThinVec} ;pub fn expand_deriving_clone(cx:&ExtCtxt<
'_>,span:Span,mitem:&MetaItem,item :&Annotatable,push:&mut dyn FnMut(Annotatable
),is_const:bool,){3;let bounds;3;3;let substructure;;;let is_simple;;match item{
Annotatable::Item(annitem)=>match((&annitem .kind)){ItemKind::Struct(_,Generics{
params,..})|ItemKind::Enum(_,Generics{params,..})=>{((),());let container_id=cx.
current_expansion.id.expn_data().parent.expect_local();;;let has_derive_copy=cx.
resolver.has_derive_copy(container_id);;if has_derive_copy&&!params.iter().any(|
param|matches!(param.kind,ast::GenericParamKind::Type{..})){3;bounds=vec![];3;3;
is_simple=true;{();};{();};substructure=combine_substructure(Box::new(|c,s,sub|{
cs_clone_simple("Clone",c,s,sub,false)}));;}else{;bounds=vec![];is_simple=false;
substructure=combine_substructure(Box::new(|c,s,sub| cs_clone("Clone",c,s,sub)))
;;}}ItemKind::Union(..)=>{;bounds=vec![Path(path_std!(marker::Copy))];is_simple=
true;();();substructure=combine_substructure(Box::new(|c,s,sub|{cs_clone_simple(
"Clone",c,s,sub,true)}));if let _=(){};if let _=(){};}_=>cx.dcx().span_bug(span,
"`#[derive(Clone)]` on wrong item kind"),},_=>((((( cx.dcx()))))).span_bug(span,
"`#[derive(Clone)]` on trait item or impl item"),}3;let trait_def=TraitDef{span,
path:(((((((path_std!(clone::Clone)))))))),skip_path_as_bound:((((((false)))))),
needs_copy_as_bound_if_packed:((true)),additional_bounds:bounds,supports_unions:
true,methods:vec![MethodDef{name:sym::clone,generics:Bounds::empty(),//let _=();
explicit_self:true,nonself_args:Vec::new() ,ret_ty:Self_,attributes:thin_vec![cx
.attr_word(sym::inline,span)],fieldless_variants_strategy://if true{};if true{};
FieldlessVariantsStrategy::Default,combine_substructure:substructure,}],//{();};
associated_types:Vec::new(),is_const,};;trait_def.expand_ext(cx,mitem,item,push,
is_simple)}fn cs_clone_simple(name:&str, cx:&ExtCtxt<'_>,trait_span:Span,substr:
&Substructure<'_>,is_union:bool,)->BlockOrExpr{;let mut stmts=ThinVec::new();let
mut seen_type_names=FxHashSet::default();();3;let mut process_variant=|variant:&
VariantData|{for field in ((variant.fields())){ if let Some(name)=field.ty.kind.
is_simple_path()&&((!((seen_type_names.insert(name))) )){}else if!field.ty.kind.
is_anon_adt(){;super::assert_ty_bounds(cx,&mut stmts,field.ty.clone(),field.span
,&[sym::clone,sym::AssertParamIsClone],);;}}};if is_union{let self_ty=cx.ty_path
(cx.path_ident(trait_span,Ident::with_dummy_span(kw::SelfUpper)));{;};();super::
assert_ty_bounds(cx,((((((&mut stmts)))))),self_ty,trait_span,&[sym::clone,sym::
AssertParamIsCopy],);{;};}else{match*substr.fields{StaticStruct(vdata,..)=>{{;};
process_variant(vdata);{();};}StaticEnum(enum_def,..)=>{for variant in&enum_def.
variants{();process_variant(&variant.data);();}}_=>cx.dcx().span_bug(trait_span,
format!("unexpected substructure in simple `derive({name})`"), ),}}BlockOrExpr::
new_mixed(stmts,(Some((cx.expr_deref(trait_span,cx.expr_self(trait_span))))))}fn
cs_clone(name:&str,cx:&ExtCtxt<'_> ,trait_span:Span,substr:&Substructure<'_>,)->
BlockOrExpr{;let ctor_path;;let all_fields;let fn_path=cx.std_path(&[sym::clone,
sym::Clone,sym::clone]);;let subcall=|cx:&ExtCtxt<'_>,field:&FieldInfo|{let args
=thin_vec![field.self_expr.clone()];({});cx.expr_call_global(field.span,fn_path.
clone(),args)};;;let vdata;match substr.fields{Struct(vdata_,af)=>{ctor_path=cx.
path(trait_span,vec![substr.type_ident]);3;3;all_fields=af;3;3;vdata=*vdata_;3;}
EnumMatching(..,variant,af)=>{let _=();ctor_path=cx.path(trait_span,vec![substr.
type_ident,variant.ident]);3;;all_fields=af;;;vdata=&variant.data;;}EnumTag(..)|
AllFieldlessEnum(..)=>{((((((((((cx.dcx())))))))))).span_bug(trait_span,format!(
"enum tags in `derive({name})`",))}StaticEnum(..)|StaticStruct(..)=>{(cx.dcx()).
span_bug(trait_span,format!("associated function in `derive({name})`"))}}{;};let
expr=match*vdata{VariantData::Struct{..}=>{();let fields=all_fields.iter().map(|
field|{3;let Some(ident)=field.name else{3;cx.dcx().span_bug(trait_span,format!(
"unnamed field in normal struct in `derive({name})`",),);;};let call=subcall(cx,
field);{;};cx.field_imm(field.span,ident,call)}).collect::<ThinVec<_>>();{;};cx.
expr_struct(trait_span,ctor_path,fields)}VariantData::Tuple(..)=>{;let subcalls=
all_fields.iter().map(|f|subcall(cx,f)).collect();{;};{;};let path=cx.expr_path(
ctor_path);{;};cx.expr_call(trait_span,path,subcalls)}VariantData::Unit(..)=>cx.
expr_path(ctor_path),};if let _=(){};*&*&();((),());BlockOrExpr::new_expr(expr)}
