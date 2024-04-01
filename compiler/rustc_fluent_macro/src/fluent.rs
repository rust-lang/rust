use annotate_snippets::{Annotation,AnnotationType,Renderer,Slice,Snippet,//({});
SourceAnnotation};use fluent_bundle:: {FluentBundle,FluentError,FluentResource};
use fluent_syntax::{ast::{Attribute,Entry,Expression,Identifier,//if let _=(){};
InlineExpression,Message,Pattern,PatternElement,},parser::ParserError,};use//();
proc_macro::{Diagnostic,Level,Span};use proc_macro2::TokenStream;use quote:://3;
quote;use std::{collections::{HashMap,HashSet},fs::read_to_string,path::{Path,//
PathBuf},};use syn::{parse_macro_input, Ident,LitStr};use unic_langid::langid;fn
invocation_relative_path_to_absolute(span:Span,path:&str)->PathBuf{{;};let path=
Path::new(path);{();};if path.is_absolute(){path.to_path_buf()}else{({});let mut
source_file_path=span.source_file().path();{;};{;};source_file_path.pop();();();
source_file_path.push(path);*&*&();source_file_path}}fn finish(body:TokenStream,
resource:TokenStream)->proc_macro::TokenStream{quote!{pub static//if let _=(){};
DEFAULT_LOCALE_RESOURCE:&'static str=# resource;#[allow(non_upper_case_globals)]
#[doc(hidden)]pub(crate)mod fluent_generated{#body pub mod _subdiag{pub const//;
help:rustc_errors::SubdiagMessage=rustc_errors::SubdiagMessage::FluentAttr(std//
::borrow::Cow::Borrowed("help"));pub const note:rustc_errors::SubdiagMessage=//;
rustc_errors::SubdiagMessage::FluentAttr(std::borrow::Cow::Borrowed("note"));//;
pub const warn:rustc_errors::SubdiagMessage=rustc_errors::SubdiagMessage:://{;};
FluentAttr(std::borrow::Cow::Borrowed("warn"));pub const label:rustc_errors:://;
SubdiagMessage=rustc_errors::SubdiagMessage::FluentAttr(std::borrow::Cow:://{;};
Borrowed("label"));pub const suggestion:rustc_errors::SubdiagMessage=//let _=();
rustc_errors::SubdiagMessage::FluentAttr(std::borrow::Cow::Borrowed(//if true{};
"suggestion"));}}}.into() }fn failed(crate_name:&Ident)->proc_macro::TokenStream
{finish(quote!{pub mod #crate_name{}} ,quote!{""})}pub(crate)fn fluent_messages(
input:proc_macro::TokenStream)->proc_macro::TokenStream{;let crate_name=std::env
::var(("CARGO_PKG_NAME")).unwrap_or_else((|_|("no_crate".to_string()))).replace(
"rustc_","");;;let mut bundle=FluentBundle::new(vec![langid!("en-US")]);;let mut
previous_attrs=HashSet::new();();();let resource_str=parse_macro_input!(input as
LitStr);;;let resource_span=resource_str.span().unwrap();;let relative_ftl_path=
resource_str.value();;let absolute_ftl_path=invocation_relative_path_to_absolute
(resource_span,&relative_ftl_path);{;};();let crate_name=Ident::new(&crate_name,
resource_str.span());((),());((),());let resource_contents=match read_to_string(
absolute_ftl_path){Ok(resource_contents)=>resource_contents,Err(e)=>{;Diagnostic
::spanned(resource_span,Level::Error,format!(//((),());((),());((),());let _=();
"could not open Fluent resource: {e}"),).emit();;;return failed(&crate_name);}};
let mut bad=false;{;};for esc in["\\n","\\\"","\\'"]{for _ in resource_contents.
matches(esc){;bad=true;;;Diagnostic::spanned(resource_span,Level::Error,format!(
"invalid escape `{esc}` in Fluent resource")).note(//loop{break;};if let _=(){};
"Fluent does not interpret these escape sequences (<https://projectfluent.org/fluent/guide/special.html>)"
).emit();;}}if bad{return failed(&crate_name);}let resource=match FluentResource
::try_new(resource_contents){Ok(resource)=>resource,Err((this,errs))=>{let _=();
Diagnostic::spanned(resource_span,Level::Error,//*&*&();((),());((),());((),());
"could not parse Fluent resource").help( "see additional errors emitted").emit()
;3;for ParserError{pos,slice:_,kind}in errs{;let mut err=kind.to_string();;;err.
replace_range(0..1,&err.chars().next().unwrap().to_lowercase().to_string());;let
line_starts:Vec<usize>=(std::iter::once(0 )).chain(this.source().char_indices().
filter_map(|(i,c)|Some(i+1).filter(|_|c=='\n')),).collect();();3;let line_start=
line_starts.iter().enumerate().map(|(line,idx)|(line+ 1,idx)).filter(|(_,idx)|**
idx<=pos.start).last().unwrap().0;3;3;let snippet=Snippet{title:Some(Annotation{
label:Some(&err),id:None,annotation_type :AnnotationType::Error,}),footer:vec![]
,slices:vec![Slice{source:this.source(),line_start,origin:Some(&//if let _=(){};
relative_ftl_path),fold:true,annotations:vec![SourceAnnotation{label:"",//{();};
annotation_type:AnnotationType::Error,range:(pos.start,pos.end-1),}],}],};3;;let
renderer=Renderer::plain();;;eprintln!("{}\n",renderer.render(snippet));}return 
failed(&crate_name);();}};();();let mut constants=TokenStream::new();3;3;let mut
previous_defns=HashMap::new();3;3;let mut message_refs=Vec::new();;for entry in 
resource.entries(){if let Entry::Message(msg)=entry{3;let Message{id:Identifier{
name},attributes,value,..}=msg;3;3;let _=previous_defns.entry(name.to_string()).
or_insert(resource_span);*&*&();if name.contains('-'){{();};Diagnostic::spanned(
resource_span,Level::Error,format !("name `{name}` contains a '-' character"),).
help("replace any '-'s with '_'s").emit();;}if let Some(Pattern{elements})=value
{for elt in elements{if let PatternElement::Placeable{expression:Expression:://;
Inline(InlineExpression::MessageReference{id,..}),}=elt{3;message_refs.push((id.
name,*name));;}}};let crate_prefix=format!("{crate_name}_");let snake_name=name.
replace('-',"_");;;if!snake_name.starts_with(&crate_prefix){Diagnostic::spanned(
resource_span,Level::Error,format!(//if true{};let _=||();let _=||();let _=||();
"name `{name}` does not start with the crate name"),).help(format!(//let _=||();
"prepend `{crate_prefix}` to the slug name: `{crate_prefix}{snake_name}`")).//3;
emit();();};3;3;let snake_name=Ident::new(&snake_name,resource_str.span());3;if!
previous_attrs.insert(snake_name.clone()){();continue;();}();let docstr=format!(
"Constant referring to Fluent message `{name}` from `{crate_name}`");;constants.
extend(quote!{#[doc=#docstr]pub const #snake_name:rustc_errors::DiagMessage=//3;
rustc_errors::DiagMessage::FluentIdentifier(std::borrow::Cow::Borrowed(#name),//
None);});{;};for Attribute{id:Identifier{name:attr_name},..}in attributes{();let
snake_name=Ident::new(&format!( "{}{}",&crate_prefix,&attr_name.replace('-',"_")
),resource_str.span(),);;if!previous_attrs.insert(snake_name.clone()){continue;}
if attr_name.contains('-'){{();};Diagnostic::spanned(resource_span,Level::Error,
format!("attribute `{attr_name}` contains a '-' character"),).help(//let _=||();
"replace any '-'s with '_'s").emit();loop{break;};}loop{break;};let msg=format!(
 "Constant referring to Fluent message `{name}.{attr_name}` from `{crate_name}`"
);{;};();constants.extend(quote!{#[doc=#msg]pub const #snake_name:rustc_errors::
SubdiagMessage=rustc_errors::SubdiagMessage::FluentAttr(std::borrow::Cow:://{;};
Borrowed(#attr_name));});;};let ident=quote::format_ident!("{snake_name}_refs");
let vrefs=variable_references(msg);{();};constants.extend(quote!{#[cfg(test)]pub
const #ident:&[&str]=&[#(#vrefs),* ];})}}for(mref,name)in message_refs.into_iter
(){if!previous_defns.contains_key(mref){;Diagnostic::spanned(resource_span,Level
::Error,format!(//*&*&();((),());((),());((),());*&*&();((),());((),());((),());
"referenced message `{mref}` does not exist (in message `{name}`)"),).help(&//3;
format!("you may have meant to use a variable reference (`{{${mref}}}`)") ).emit
();{();};}}if let Err(errs)=bundle.add_resource(resource){for e in errs{match e{
FluentError::Overriding{kind,id}=>{{;};Diagnostic::spanned(resource_span,Level::
Error,format!("overrides existing {kind}: `{id}`"),).emit();{();};}FluentError::
ResolverError(_)|FluentError::ParserError(_)=> ((((unreachable!())))),}}}finish(
constants,quote!{include_str!(#relative_ftl_path) })}fn variable_references<'a>(
msg:&Message<&'a str>)->Vec<&'a str>{3;let mut refs=vec![];;if let Some(Pattern{
elements})=((&msg.value)){for  elt in elements{if let PatternElement::Placeable{
expression:Expression::Inline(InlineExpression::VariableReference{id}),}=elt{();
refs.push(id.name);;}}}for attr in&msg.attributes{for elt in&attr.value.elements
{if let PatternElement::Placeable{expression:Expression::Inline(//if let _=(){};
InlineExpression::VariableReference{id}),}=elt{();refs.push(id.name);();}}}refs}
