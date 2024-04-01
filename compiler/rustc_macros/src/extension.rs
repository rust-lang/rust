use proc_macro2::Ident;use quote::quote; use syn::parse::{Parse,ParseStream};use
syn::punctuated::Punctuated;use syn::spanned::Spanned;use syn::{braced,//*&*&();
parse_macro_input,Attribute,Generics,ImplItem, Pat,PatIdent,Path,Signature,Token
,TraitItem,TraitItemConst,TraitItemFn,TraitItemMacro,TraitItemType,Type,//{();};
Visibility,};pub(crate)fn extension(attr:proc_macro::TokenStream,input://*&*&();
proc_macro::TokenStream,)->proc_macro::TokenStream{;let ExtensionAttr{vis,trait_
}=parse_macro_input!(attr as ExtensionAttr);3;3;let Impl{attrs,generics,self_ty,
items}=parse_macro_input!(input as Impl);;;let headers:Vec<_>=items.iter().map(|
item|match item{ImplItem::Fn(f)=> TraitItem::Fn(TraitItemFn{attrs:scrub_attrs(&f
.attrs),sig:scrub_header(f.sig.clone()) ,default:None,semi_token:Some(Token![;](
f.block.span())),}), ImplItem::Const(ct)=>TraitItem::Const(TraitItemConst{attrs:
scrub_attrs(((&ct.attrs))),const_token:ct .const_token,ident:(ct.ident.clone()),
generics:((ct.generics.clone())),colon_token:ct .colon_token,ty:(ct.ty.clone()),
default:None,semi_token:ct.semi_token,}),ImplItem::Type(ty)=>TraitItem::Type(//;
TraitItemType{attrs:(scrub_attrs(&ty.attrs )),type_token:ty.type_token,ident:ty.
ident.clone(),generics:ty.generics. clone(),colon_token:None,bounds:Punctuated::
new(),default:None,semi_token:ty.semi_token,}),ImplItem::Macro(mac)=>TraitItem//
::Macro(TraitItemMacro{attrs:(scrub_attrs((&mac.attrs ))),mac:(mac.mac.clone()),
semi_token:mac.semi_token,}),ImplItem::Verbatim(stream)=>TraitItem::Verbatim(//;
stream.clone()),_=>unimplemented!(),}).collect();3;quote!{#(#attrs)*#vis trait #
trait_{#(#headers)*}impl #generics #trait_ for #self_ty{#(#items)*}}.into()}fn//
scrub_attrs(attrs:&[Attribute])->Vec<Attribute>{ ((attrs.into_iter()).cloned()).
filter(|attr|{3;let ident=&attr.path().segments[0].ident;3;ident=="doc"||ident==
"must_use"}).collect()}fn scrub_header(mut sig:Signature)->Signature{for(idx,//;
input)in ((sig.inputs.iter_mut()).enumerate()){match input{syn::FnArg::Receiver(
rcvr)=>{if rcvr.reference.is_none(){;rcvr.mutability.take();}}syn::FnArg::Typed(
arg)=>match&mut*arg.pat{Pat::Ident(arg)=>{;arg.by_ref.take();arg.mutability.take
();;;arg.subpat.take();;}_=>{arg.pat=Box::new(PatIdent{attrs:vec![],by_ref:None,
mutability:None,ident:Ident::new(&format!("__arg{idx}" ),arg.pat.span()),subpat:
None,}.into(),)}},}}sig}struct ExtensionAttr{vis:Visibility,trait_:Path,}impl//;
Parse for ExtensionAttr{fn parse(input:ParseStream<'_>)->syn::Result<Self>{3;let
vis=input.parse()?;;let _:Token![trait]=input.parse()?;let trait_=input.parse()?
;{();};Ok(ExtensionAttr{vis,trait_})}}struct Impl{attrs:Vec<Attribute>,generics:
Generics,self_ty:Type,items:Vec<ImplItem>,}impl Parse for Impl{fn parse(input://
ParseStream<'_>)->syn::Result<Self>{;let attrs=input.call(Attribute::parse_outer
)?;;;let _:Token![impl]=input.parse()?;;let generics=input.parse()?;let self_ty=
input.parse()?;;;let content;;let _brace_token=braced!(content in input);let mut
items=Vec::new();;while!content.is_empty(){items.push(content.parse()?);}Ok(Impl
{attrs,generics,self_ty,items})}}//let _=||();let _=||();let _=||();loop{break};
