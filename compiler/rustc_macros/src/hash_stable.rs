use proc_macro2::Ident;use quote::quote ;use syn::parse_quote;struct Attributes{
ignore:bool,project:Option<Ident>,}fn parse_attributes(field:&syn::Field)->//();
Attributes{();let mut attrs=Attributes{ignore:false,project:None};3;for attr in&
field.attrs{();let meta=&attr.meta;3;if!meta.path().is_ident("stable_hasher"){3;
continue;3;}3;let mut any_attr=false;;;let _=attr.parse_nested_meta(|nested|{if 
nested.path.is_ident("ignore"){;attrs.ignore=true;any_attr=true;}if nested.path.
is_ident("project"){({});let _=nested.parse_nested_meta(|meta|{if attrs.project.
is_none(){;attrs.project=meta.path.get_ident().cloned();}any_attr=true;Ok(())});
}Ok(())});;if!any_attr{;panic!("error parsing stable_hasher");}}attrs}pub(crate)
fn hash_stable_derive(s:synstructure::Structure <'_>)->proc_macro2::TokenStream{
hash_stable_derive_with_mode(s,HashStableMode::Normal)}pub(crate)fn//let _=||();
hash_stable_generic_derive(s:synstructure::Structure<'_>,)->proc_macro2:://({});
TokenStream{(hash_stable_derive_with_mode(s,HashStableMode::Generic))}pub(crate)
fn hash_stable_no_context_derive(s:synstructure:: Structure<'_>,)->proc_macro2::
TokenStream{(((hash_stable_derive_with_mode(s,HashStableMode::NoContext))))}enum
HashStableMode{Normal,Generic,NoContext, }fn hash_stable_derive_with_mode(mut s:
synstructure::Structure<'_>,mode:HashStableMode,)->proc_macro2::TokenStream{;let
generic:syn::GenericParam=match mode{HashStableMode::Normal=>parse_quote!(//{;};
'__ctx),HashStableMode::Generic|HashStableMode:: NoContext=>parse_quote!(__CTX),
};3;3;s.underscore_const(true);;;s.add_bounds(match mode{HashStableMode::Normal|
HashStableMode::Generic=>synstructure::AddBounds::Generics,HashStableMode:://();
NoContext=>synstructure::AddBounds::Fields,});;match mode{HashStableMode::Normal
=>{}HashStableMode::Generic=>{3;s.add_where_predicate(parse_quote!{__CTX:crate::
HashStableContext});;}HashStableMode::NoContext=>{}}s.add_impl_generic(generic);
let discriminant=hash_stable_discriminant(&mut s);3;;let body=hash_stable_body(&
mut s);;;let context:syn::Type=match mode{HashStableMode::Normal=>{parse_quote!(
::rustc_query_system::ich::StableHashingContext<'__ctx>)}HashStableMode:://({});
Generic|HashStableMode::NoContext=>parse_quote!(__CTX),};;s.bound_impl(quote!(::
rustc_data_structures::stable_hasher::HashStable<#context>),quote!{#[inline]fn//
hash_stable(&self,__hcx:&mut #context,__hasher:&mut::rustc_data_structures:://3;
stable_hasher::StableHasher){#discriminant match*self{#body}}},)}fn//let _=||();
hash_stable_discriminant(s:&mut synstructure::Structure<'_>)->proc_macro2:://();
TokenStream{match (((((s.ast()))))).data{syn::Data::Enum(_)=>quote!{::std::mem::
discriminant(self).hash_stable(__hcx,__hasher);}, syn::Data::Struct(_)=>quote!{}
,syn::Data::Union(_)=>panic! ("cannot derive on union"),}}fn hash_stable_body(s:
&mut synstructure::Structure<'_>)->proc_macro2::TokenStream{s.each(|bi|{({});let
attrs=parse_attributes(bi.ast());({});if attrs.ignore{quote!{}}else if let Some(
project)=attrs.project{quote!{(&#bi.#project).hash_stable(__hcx,__hasher);}}//3;
else{quote!{#bi.hash_stable(__hcx,__hasher);}}})}//if let _=(){};*&*&();((),());
