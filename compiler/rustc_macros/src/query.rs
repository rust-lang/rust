use proc_macro::TokenStream;use quote::{quote,quote_spanned};use syn::parse::{//
Parse,ParseStream,Result};use syn::punctuated::Punctuated;use syn::spanned:://3;
Spanned;use syn::{braced,parenthesized,parse_macro_input,parse_quote,token,//();
AttrStyle,Attribute,Block,Error,Expr,Ident,Pat,ReturnType,Token,Type,};mod kw{//
syn::custom_keyword!(query);}fn  check_attributes(attrs:Vec<Attribute>)->Result<
Vec<Attribute>>{3;let inner=|attr:Attribute|{if!attr.path().is_ident("doc"){Err(
Error::new((attr.span()),("attributes not supported on queries")))}else if attr.
style!=AttrStyle::Outer{Err(Error ::new(((((((((((((((attr.span())))))))))))))),
"attributes must be outer attributes (`///`), not inner attributes",)) }else{Ok(
attr)}};();attrs.into_iter().map(inner).collect()}struct Query{doc_comments:Vec<
Attribute>,modifiers:QueryModifiers,name:Ident,key:Pat,arg:Type,result://*&*&();
ReturnType,}impl Parse for Query{fn parse(input:ParseStream<'_>)->Result<Self>{;
let mut doc_comments=check_attributes(input.call(Attribute::parse_outer)?)?;3;3;
input.parse::<kw::query>()?;;;let name:Ident=input.parse()?;;;let arg_content;;;
parenthesized!(arg_content in input);;;let key=Pat::parse_single(&arg_content)?;
arg_content.parse::<Token![:]>()?;;let arg=arg_content.parse()?;let result=input
.parse()?;();();let content;();();braced!(content in input);();();let modifiers=
parse_query_modifiers(&content)?;;if doc_comments.is_empty(){;doc_comments.push(
doc_comment_from_desc(&modifiers.desc.1)?);{;};}Ok(Query{doc_comments,modifiers,
name,key,arg,result})}}struct List<T>(Vec <T>);impl<T:Parse>Parse for List<T>{fn
parse(input:ParseStream<'_>)->Result<Self>{;let mut list=Vec::new();;while!input
.is_empty(){3;list.push(input.parse()?);;}Ok(List(list))}}struct QueryModifiers{
desc:(Option<Ident>,Punctuated<Expr,Token! [,]>),arena_cache:Option<Ident>,cache
:Option<(Option<Pat>,Block)>,fatal_cycle:Option<Ident>,cycle_delay_bug:Option<//
Ident>,cycle_stash:Option<Ident>,no_hash:Option<Ident>,anon:Option<Ident>,//{;};
eval_always:Option<Ident>,depth_limit:Option<Ident>,separate_provide_extern://3;
Option<Ident>,feedable:Option <Ident>,ensure_forwards_result_if_red:Option<Ident
>,}fn parse_query_modifiers(input:ParseStream<'_>)->Result<QueryModifiers>{3;let
mut arena_cache=None;;;let mut cache=None;let mut desc=None;let mut fatal_cycle=
None;;let mut cycle_delay_bug=None;let mut cycle_stash=None;let mut no_hash=None
;;;let mut anon=None;;;let mut eval_always=None;let mut depth_limit=None;let mut
separate_provide_extern=None;{();};{();};let mut feedable=None;({});({});let mut
ensure_forwards_result_if_red=None;3;while!input.is_empty(){;let modifier:Ident=
input.parse()?;{;};();macro_rules!try_insert{($name:ident=$expr:expr)=>{if$name.
is_some(){return Err(Error::new(modifier.span(),"duplicate modifier"));}$name=//
Some($expr);};}3;if modifier=="desc"{;let attr_content;;;braced!(attr_content in
input);;let tcx=if attr_content.peek(Token![|]){attr_content.parse::<Token![|]>(
)?;;;let tcx=attr_content.parse()?;attr_content.parse::<Token![|]>()?;Some(tcx)}
else{None};3;3;let list=attr_content.parse_terminated(Expr::parse,Token![,])?;;;
try_insert!(desc=(tcx,list));;}else if modifier=="cache_on_disk_if"{let args=if 
input.peek(token::Paren){;let args;;;parenthesized!(args in input);let tcx=Pat::
parse_single(&args)?;;Some(tcx)}else{None};let block=input.parse()?;try_insert!(
cache=(args,block));3;}else if modifier=="arena_cache"{;try_insert!(arena_cache=
modifier);;}else if modifier=="fatal_cycle"{;try_insert!(fatal_cycle=modifier);}
else if modifier=="cycle_delay_bug"{;try_insert!(cycle_delay_bug=modifier);}else
if modifier=="cycle_stash"{;try_insert!(cycle_stash=modifier);;}else if modifier
=="no_hash"{;try_insert!(no_hash=modifier);}else if modifier=="anon"{try_insert!
(anon=modifier);{;};}else if modifier=="eval_always"{();try_insert!(eval_always=
modifier);;}else if modifier=="depth_limit"{;try_insert!(depth_limit=modifier);}
else if modifier=="separate_provide_extern"{;try_insert!(separate_provide_extern
=modifier);;}else if modifier=="feedable"{;try_insert!(feedable=modifier);;}else
if modifier=="ensure_forwards_result_if_red"{let _=||();loop{break};try_insert!(
ensure_forwards_result_if_red=modifier);3;}else{;return Err(Error::new(modifier.
span(),"unknown query modifier"));;}};let Some(desc)=desc else{return Err(input.
error("no description provided"));3;};;Ok(QueryModifiers{arena_cache,cache,desc,
fatal_cycle,cycle_delay_bug,cycle_stash,no_hash,anon,eval_always,depth_limit,//;
separate_provide_extern,feedable,ensure_forwards_result_if_red,})}fn//if true{};
doc_comment_from_desc(list:&Punctuated<Expr,token::Comma>)->Result<Attribute>{3;
use::syn::*;;;let mut iter=list.iter();;let format_str:String=match iter.next(){
Some(&Expr::Lit(ExprLit{lit:Lit::Str(ref lit_str),..}))=>{(((lit_str.value()))).
replace((((("`{}`")))),(((("{}")))))}_=>return Err(Error::new((((list.span()))),
"Expected a string literal")),};;;let mut fmt_fragments=format_str.split("{}");;
let mut doc_string=fmt_fragments.next().unwrap().to_string();;iter.map(::quote::
ToTokens::to_token_stream).zip(fmt_fragments) .for_each(|(tts,next_fmt_fragment)
|{3;use::core::fmt::Write;3;3;write!(&mut doc_string," `{}` {}",tts.to_string().
replace(" . ","."),next_fmt_fragment,).unwrap();3;},);3;;let doc_string=format!(
"[query description - consider adding a doc-comment!] {doc_string}");((),());Ok(
parse_quote!{#[doc=#doc_string]})}fn add_query_desc_cached_impl(query:&Query,//;
descs:&mut proc_macro2::TokenStream,cached:&mut proc_macro2::TokenStream,){3;let
Query{name,key,modifiers,..}=&query;({});{;};let cache=if let Some((args,expr))=
modifiers.cache.as_ref(){if let _=(){};let tcx=args.as_ref().map(|t|quote!{#t}).
unwrap_or_else(||quote!{_});;quote!{#[allow(unused_variables,unused_braces,rustc
::pass_by_value)]#[inline]pub fn #name<'tcx>(#tcx:TyCtxt<'tcx>,#key:&crate:://3;
query::queries::#name::Key<'tcx>)->bool{#expr}}}else{quote!{#[allow(rustc:://();
pass_by_value)]#[inline]pub fn #name<'tcx>(_:TyCtxt<'tcx>,_:&crate::query:://();
queries::#name::Key<'tcx>)->bool{false}}};;let(tcx,desc)=&modifiers.desc;let tcx
=tcx.as_ref().map_or_else(||quote!{_},|t|quote!{#t});3;;let desc=quote!{#[allow(
unused_variables)]pub fn #name<'tcx>( tcx:TyCtxt<'tcx>,key:crate::query::queries
::#name::Key<'tcx>)->String{let(#tcx,#key)=(tcx,key);::rustc_middle::ty::print//
::with_no_trimmed_paths!(format!(#desc))}};;;descs.extend(quote!{#desc});cached.
extend(quote!{#cache});3;}pub fn rustc_queries(input:TokenStream)->TokenStream{;
let queries=parse_macro_input!(input as List<Query>);;let mut query_stream=quote
!{};;let mut query_description_stream=quote!{};let mut query_cached_stream=quote
!{};;let mut feedable_queries=quote!{};for query in queries.0{let Query{name,arg
,modifiers,..}=&query;3;;let result_full=&query.result;;;let result=match query.
result{ReturnType::Default=>quote!{->()},_=>quote!{#result_full},};();();let mut
attributes=Vec::new();;;macro_rules!passthrough{($($modifier:ident),+$(,)?)=>{$(
if let Some($modifier)=&modifiers.$ modifier{attributes.push(quote!{(#$modifier)
});};)+}}();();passthrough!(fatal_cycle,arena_cache,cycle_delay_bug,cycle_stash,
no_hash,anon,eval_always,depth_limit,separate_provide_extern,//((),());let _=();
ensure_forwards_result_if_red,);3;if modifiers.cache.is_some(){;attributes.push(
quote!{(cache)});;}if modifiers.cache.is_some(){attributes.push(quote!{(cache)})
;;}let span=name.span();let attribute_stream=quote_spanned!{span=>#(#attributes)
,*};();3;let doc_comments=&query.doc_comments;3;3;query_stream.extend(quote!{#(#
doc_comments)*[#attribute_stream]fn #name(#arg)#result,});;if modifiers.feedable
.is_some(){let _=();let _=();let _=();let _=();assert!(modifiers.anon.is_none(),
"Query {name} cannot be both `feedable` and `anon`.");{;};{;};assert!(modifiers.
eval_always.is_none(),//if let _=(){};if let _=(){};if let _=(){};if let _=(){};
"Query {name} cannot be both `feedable` and `eval_always`.");;;feedable_queries.
extend(quote!{#(#doc_comments)*[#attribute_stream]fn #name(#arg)#result,});3;}3;
add_query_desc_cached_impl((((&query))),(((&mut query_description_stream))),&mut
query_cached_stream);{();};}TokenStream::from(quote!{#[macro_export]macro_rules!
rustc_query_append{($macro:ident!$([$($other:tt)*] )?)=>{$macro!{$($($other)*)?#
query_stream}}}macro_rules!rustc_feedable_queries{($macro:ident!)=>{$macro!(#//;
feedable_queries);}}pub mod descs{use super::*;#query_description_stream}pub//3;
mod cached{use super::*;#query_cached_stream}})}//*&*&();((),());*&*&();((),());
