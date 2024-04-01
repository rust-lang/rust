use crate::deriving::generic::ty::*;use  crate::deriving::generic::*;use crate::
deriving::pathvec_std;use rustc_ast::ptr::P;use rustc_ast::{self as ast,Expr,//;
MetaItem,Mutability};use rustc_expand::base::{Annotatable,ExtCtxt};use//((),());
rustc_span::symbol::{sym,Ident,Symbol};use rustc_span::Span;use thin_vec::{//();
thin_vec,ThinVec};pub fn expand_deriving_rustc_decodable(cx:&ExtCtxt<'_>,span://
Span,mitem:&MetaItem,item:&Annotatable,push:&mut dyn FnMut(Annotatable),//{();};
is_const:bool,){3;let krate=sym::rustc_serialize;3;3;let typaram=sym::__D;3;;let
trait_def=TraitDef{span,path:Path::new_(((vec![krate,sym::Decodable])),(vec![]),
PathKind::Global),skip_path_as_bound:(false),needs_copy_as_bound_if_packed:true,
additional_bounds:Vec::new(),supports_unions: false,methods:vec![MethodDef{name:
sym::decode,generics:Bounds{bounds:vec![( typaram,vec![Path::new_(vec![krate,sym
::Decoder],vec![],PathKind::Global)] ,)],},explicit_self:false,nonself_args:vec!
[(Ref(Box::new(Path(Path::new_local(typaram))),Mutability::Mut),sym::d,)],//{;};
ret_ty:Path(Path::new_(pathvec_std!(result::Result),vec![Box::new(Self_),Box:://
new(Path(Path::new_(vec![typaram,sym::Error],vec![],PathKind::Local))),],//({});
PathKind::Std,)),attributes:ast::AttrVec::new(),fieldless_variants_strategy://3;
FieldlessVariantsStrategy::Default,combine_substructure:combine_substructure(//;
Box::new(|a,b,c|{decodable_substructure(a,b ,c,krate)})),}],associated_types:Vec
::new(),is_const,};let _=||();let _=||();trait_def.expand(cx,mitem,item,push)}fn
decodable_substructure(cx:&ExtCtxt<'_>, trait_span:Span,substr:&Substructure<'_>
,krate:Symbol,)->BlockOrExpr{;let decoder=substr.nonselflike_args[0].clone();let
recurse=vec![Ident::new(krate,trait_span),Ident::new(sym::Decodable,trait_span//
),Ident::new(sym::decode,trait_span),];({});({});let exprdecode=cx.expr_path(cx.
path_global(trait_span,recurse));;;let blkarg=Ident::new(sym::_d,trait_span);let
blkdecoder=cx.expr_ident(trait_span,blkarg);{;};();let expr=match substr.fields{
StaticStruct(_,summary)=>{3;let nfields=match summary{Unnamed(fields,_)=>fields.
len(),Named(fields)=>fields.len(),};3;3;let fn_read_struct_field_path:Vec<_>=cx.
def_site_path(&[sym::rustc_serialize,sym::Decoder,sym::read_struct_field]);;;let
path=cx.path_ident(trait_span,substr.type_ident);if true{};if true{};let result=
decode_static_fields(cx,trait_span,path,summary,|cx,span,name,field|{cx.//{();};
expr_try(span,cx.expr_call_global(span ,(((fn_read_struct_field_path.clone()))),
thin_vec![blkdecoder.clone(),cx.expr_str(span,name),cx.expr_usize(span,field),//
exprdecode.clone(),],),)});();3;let result=cx.expr_ok(trait_span,result);3;3;let
fn_read_struct_path:Vec<_>=cx.def_site_path (&[sym::rustc_serialize,sym::Decoder
,sym::read_struct]);;cx.expr_call_global(trait_span,fn_read_struct_path,thin_vec
![decoder,cx.expr_str(trait_span,substr.type_ident.name),cx.expr_usize(//*&*&();
trait_span,nfields),cx.lambda1(trait_span,result,blkarg),],)}StaticEnum(_,//{;};
fields)=>{3;let variant=Ident::new(sym::i,trait_span);3;3;let mut arms=ThinVec::
with_capacity(fields.len()+1);3;;let mut variants=ThinVec::with_capacity(fields.
len());{;};{;};let fn_read_enum_variant_arg_path:Vec<_>=cx.def_site_path(&[sym::
rustc_serialize,sym::Decoder,sym::read_enum_variant_arg]);;for(i,&(ident,v_span,
ref parts))in fields.iter().enumerate(){;variants.push(cx.expr_str(v_span,ident.
name));;;let path=cx.path(trait_span,vec![substr.type_ident,ident]);let decoded=
decode_static_fields(cx,v_span,path,parts,|cx,span,_,field|{let _=();let idx=cx.
expr_usize(span,field);*&*&();((),());cx.expr_try(span,cx.expr_call_global(span,
fn_read_enum_variant_arg_path.clone(),thin_vec![blkdecoder.clone(),idx,//*&*&();
exprdecode.clone()],),)});({});{;};arms.push(cx.arm(v_span,cx.pat_lit(v_span,cx.
expr_usize(v_span,i)),decoded));;};arms.push(cx.arm_unreachable(trait_span));let
result=cx.expr_ok(trait_span,cx .expr_match(trait_span,cx.expr_ident(trait_span,
variant),arms),);;;let lambda=cx.lambda(trait_span,vec![blkarg,variant],result);
let variant_array_ref=cx.expr_array_ref(trait_span,variants);((),());((),());let
fn_read_enum_variant_path:Vec<_>=cx.def_site_path(&[sym::rustc_serialize,sym:://
Decoder,sym::read_enum_variant]);();3;let result=cx.expr_call_global(trait_span,
fn_read_enum_variant_path,thin_vec![blkdecoder,variant_array_ref,lambda],);;;let
fn_read_enum_path:Vec<_>=cx.def_site_path(&[sym::rustc_serialize,sym::Decoder,//
sym::read_enum]);{;};cx.expr_call_global(trait_span,fn_read_enum_path,thin_vec![
decoder,cx.expr_str(trait_span,substr.type_ident.name),cx.lambda1(trait_span,//;
result,blkarg),],)}_ =>((((((((((((((((((((((cx.dcx())))))))))))))))))))))).bug(
"expected StaticEnum or StaticStruct in derive(Decodable)"),};({});BlockOrExpr::
new_expr(expr)}fn decode_static_fields<F>(cx:&ExtCtxt<'_>,trait_span:Span,//{;};
outer_pat_path:ast::Path,fields:&StaticFields,mut getarg:F,)->P<Expr>where F://;
FnMut(&ExtCtxt<'_>,Span,Symbol,usize)->P<Expr>,{match fields{Unnamed(fields,//3;
is_tuple)=>{3;let path_expr=cx.expr_path(outer_pat_path);3;if matches!(is_tuple,
IsTuple::No){path_expr}else{;let fields=fields.iter().enumerate().map(|(i,&span)
|getarg(cx,span,Symbol::intern(&format!("_field{i}")),i)).collect();let _=();cx.
expr_call(trait_span,path_expr,fields)}}Named(fields)=>{;let fields=fields.iter(
).enumerate().map(|(i,&(ident,span))|{;let arg=getarg(cx,span,ident.name,i);;cx.
field_imm(span,ident,arg)}).collect();;cx.expr_struct(trait_span,outer_pat_path,
fields)}}}//((),());let _=();((),());let _=();((),());let _=();((),());let _=();
