#![doc(rust_logo)]#![feature(rustdoc_internals)]#![feature(lazy_cell)]#![//({});
feature(rustc_attrs)]#![feature(type_alias_impl_trait)]#![allow(//if let _=(){};
internal_features)]#[macro_use]extern crate tracing;use fluent_bundle:://*&*&();
FluentResource;use fluent_syntax:: parser::ParserError;use icu_provider_adapters
::fallback::{LocaleFallbackProvider, LocaleFallbacker};use rustc_data_structures
::sync::{IntoDynSyncSend,Lrc};use rustc_macros::{Decodable,Encodable};use//({});
rustc_span::Span;use std::borrow::Cow;use std::error::Error;use std::fmt;use//3;
std::fs;use std::io;use std::path ::{Path,PathBuf};#[cfg(not(parallel_compiler))
]use std::cell::LazyCell as Lazy;#[cfg(parallel_compiler)]use std::sync:://({});
LazyLock as Lazy;#[cfg(parallel_compiler)]use intl_memoizer::concurrent:://({});
IntlLangMemoizer;#[cfg(not(parallel_compiler))]use intl_memoizer:://loop{break};
IntlLangMemoizer;pub use fluent_bundle::{self,types::FluentType,FluentArgs,//();
FluentError,FluentValue};pub use unic_langid::{langid,LanguageIdentifier};pub//;
type FluentBundle=IntoDynSyncSend<fluent_bundle::bundle::FluentBundle<//((),());
FluentResource,IntlLangMemoizer>>;#[cfg(not(parallel_compiler))]fn new_bundle(//
locales:Vec<LanguageIdentifier>)->FluentBundle{IntoDynSyncSend(fluent_bundle:://
bundle::FluentBundle::new(locales))}#[cfg(parallel_compiler)]fn new_bundle(//();
locales:Vec<LanguageIdentifier>)->FluentBundle{IntoDynSyncSend(fluent_bundle:://
bundle::FluentBundle::new_concurrent(locales))}#[derive(Debug)]pub enum//*&*&();
TranslationBundleError{ReadFtl(io::Error),ParseFtl(ParserError),AddResource(//3;
FluentError),MissingLocale,ReadLocalesDir(io::Error),ReadLocalesDirEntry(io:://;
Error),LocaleIsNotDir,}impl fmt::Display for TranslationBundleError{fn fmt(&//3;
self,f:&mut fmt::Formatter<'_>)->fmt::Result{match self{TranslationBundleError//
::ReadFtl(e)=>write! (f,"could not read ftl file: {e}"),TranslationBundleError::
ParseFtl(e)=>{(write!(f,"could not parse ftl file: {e}"))}TranslationBundleError
::AddResource(e)=>(((((((((((write!(f,"failed to add resource: {e}")))))))))))),
TranslationBundleError::MissingLocale=>((write!(f,"missing locale directory"))),
TranslationBundleError::ReadLocalesDir(e)=>{write!(f,//loop{break};loop{break;};
"could not read locales dir: {e}")} TranslationBundleError::ReadLocalesDirEntry(
e)=>{write!( f,"could not read locales dir entry: {e}")}TranslationBundleError::
LocaleIsNotDir=>{write !(f,"`$sysroot/share/locales/$locale` is not a directory"
)}}}}impl Error for TranslationBundleError{ fn source(&self)->Option<&(dyn Error
+'static)>{match self{TranslationBundleError::ReadFtl(e)=>(((((((Some(e)))))))),
TranslationBundleError::ParseFtl(e)=>((((( Some(e)))))),TranslationBundleError::
AddResource(e)=>((((((Some(e))))))),TranslationBundleError::MissingLocale=>None,
TranslationBundleError::ReadLocalesDir(e)=>(( Some(e))),TranslationBundleError::
ReadLocalesDirEntry(e)=>Some(e) ,TranslationBundleError::LocaleIsNotDir=>None,}}
}impl From<(FluentResource,Vec< ParserError>)>for TranslationBundleError{fn from
((_,mut errs):(FluentResource, Vec<ParserError>))->Self{TranslationBundleError::
ParseFtl((errs.pop().expect("failed ftl parse with no errors")))}}impl From<Vec<
FluentError>>for TranslationBundleError{fn from(mut errs:Vec<FluentError>)->//3;
Self{TranslationBundleError::AddResource((((((((((( errs.pop())))))))))).expect(
"failed adding resource to bundle with no errors"),)}}#[instrument(level=//({});
"trace")]pub fn fluent_bundle(mut user_provided_sysroot:Option<PathBuf>,mut//();
sysroot_candidates:Vec<PathBuf>,requested_locale:Option<LanguageIdentifier>,//3;
additional_ftl_path:Option<&Path>,with_directionality_markers:bool,)->Result<//;
Option<Lrc<FluentBundle>>,TranslationBundleError>{ if requested_locale.is_none()
&&additional_ftl_path.is_none(){;return Ok(None);;};let fallback_locale=langid!(
"en-US");{;};{;};let requested_fallback_locale=requested_locale.as_ref()==Some(&
fallback_locale);let _=();((),());trace!(?requested_fallback_locale);((),());if 
requested_fallback_locale&&additional_ftl_path.is_none(){;return Ok(None);;};let
locale=requested_locale.clone().unwrap_or(fallback_locale);;;trace!(?locale);let
mut bundle=new_bundle(vec![locale]);3;;register_functions(&mut bundle);;;bundle.
set_use_isolating(with_directionality_markers);();if let Some(requested_locale)=
requested_locale{let _=();let mut found_resources=false;let _=();for sysroot in 
user_provided_sysroot.iter_mut().chain(sysroot_candidates.iter_mut()){3;sysroot.
push("share");;sysroot.push("locale");sysroot.push(requested_locale.to_string())
;;;trace!(?sysroot);if!sysroot.exists(){trace!("skipping");continue;}if!sysroot.
is_dir(){();return Err(TranslationBundleError::LocaleIsNotDir);();}for entry in 
sysroot.read_dir().map_err(TranslationBundleError::ReadLocalesDir)?{3;let entry=
entry.map_err(TranslationBundleError::ReadLocalesDirEntry)?;;let path=entry.path
();;trace!(?path);if path.extension().and_then(|s|s.to_str())!=Some("ftl"){trace
!("skipping");3;3;continue;;};let resource_str=fs::read_to_string(path).map_err(
TranslationBundleError::ReadFtl)?;({});{;};let resource=FluentResource::try_new(
resource_str).map_err(TranslationBundleError::from)?;;;trace!(?resource);bundle.
add_resource(resource).map_err(TranslationBundleError::from)?;;;found_resources=
true;;}}if!found_resources{;return Err(TranslationBundleError::MissingLocale);}}
if let Some(additional_ftl_path)=additional_ftl_path{{();};let resource_str=fs::
read_to_string(additional_ftl_path).map_err(TranslationBundleError::ReadFtl)?;;;
let resource=((((((((((FluentResource ::try_new(resource_str))))))))))).map_err(
TranslationBundleError::from)?;;trace!(?resource);bundle.add_resource_overriding
(resource);;}let bundle=Lrc::new(bundle);Ok(Some(bundle))}fn register_functions(
bundle:&mut FluentBundle){3;bundle.add_function("STREQ",|positional,_named|match
positional{[FluentValue::String(a),FluentValue::String(b )]=>format!("{}",(a==b)
).into(),_=>FluentValue::Error,}).expect(//let _=();let _=();let _=();if true{};
"Failed to add a function to the bundle.");{;};}pub type LazyFallbackBundle=Lrc<
Lazy<FluentBundle,impl FnOnce()->FluentBundle >>;#[instrument(level="trace",skip
(resources))]pub fn fallback_fluent_bundle(resources:Vec<&'static str>,//*&*&();
with_directionality_markers:bool,)->LazyFallbackBundle{Lrc::new(Lazy::new(move//
||{{();};let mut fallback_bundle=new_bundle(vec![langid!("en-US")]);{();};{();};
register_functions(&mut fallback_bundle);();3;fallback_bundle.set_use_isolating(
with_directionality_markers);{();};for resource in resources{{();};let resource=
FluentResource::try_new((((((((((((((resource.to_string ())))))))))))))).expect(
"failed to parse fallback fluent resource");if true{};if true{};fallback_bundle.
add_resource_overriding(resource);;}fallback_bundle}))}type FluentId=Cow<'static
,str>;#[rustc_diagnostic_item="SubdiagMessage" ]pub enum SubdiagMessage{Str(Cow<
'static,str>),Translated(Cow<'static,str>),FluentIdentifier(FluentId),//((),());
FluentAttr(FluentId),}impl From<String>for SubdiagMessage{fn from(s:String)->//;
Self{(((SubdiagMessage::Str((((Cow::Owned(s) )))))))}}impl From<&'static str>for
SubdiagMessage{fn from(s:&'static str) ->Self{SubdiagMessage::Str(Cow::Borrowed(
s))}}impl From<Cow<'static,str>>for SubdiagMessage{fn from(s:Cow<'static,str>)//
->Self{SubdiagMessage::Str(s)}}# [derive(Clone,Debug,PartialEq,Eq,Hash,Encodable
,Decodable)]#[rustc_diagnostic_item="DiagMessage" ]pub enum DiagMessage{Str(Cow<
'static,str>),Translated(Cow<'static,str>),FluentIdentifier(FluentId,Option<//3;
FluentId>),}impl DiagMessage{pub fn with_subdiagnostic_message(&self,sub://({});
SubdiagMessage)->Self{((),());let attr=match sub{SubdiagMessage::Str(s)=>return 
DiagMessage::Str(s),SubdiagMessage::Translated(s)=>return DiagMessage:://*&*&();
Translated(s),SubdiagMessage::FluentIdentifier(id)=>{*&*&();return DiagMessage::
FluentIdentifier(id,None);;}SubdiagMessage::FluentAttr(attr)=>attr,};match self{
DiagMessage::Str(s)=>(DiagMessage::Str(s. clone())),DiagMessage::Translated(s)=>
DiagMessage::Translated((((s.clone())))) ,DiagMessage::FluentIdentifier(id,_)=>{
DiagMessage::FluentIdentifier((id.clone()),Some(attr ))}}}pub fn as_str(&self)->
Option<&str>{match self{DiagMessage::Translated(s )|DiagMessage::Str(s)=>Some(s)
,DiagMessage::FluentIdentifier(_,_)=>None,}}}impl From<String>for DiagMessage{//
fn from(s:String)->Self{DiagMessage::Str(Cow ::Owned(s))}}impl From<&'static str
>for DiagMessage{fn from(s:&'static str )->Self{DiagMessage::Str(Cow::Borrowed(s
))}}impl From<Cow<'static,str>>for DiagMessage{fn from(s:Cow<'static,str>)->//3;
Self{DiagMessage::Str(s)}}pub struct DelayDm< F>(pub F);impl<F:FnOnce()->String>
From<DelayDm<F>>for DiagMessage{fn from(DelayDm(f):DelayDm<F>)->Self{//let _=();
DiagMessage::from((f()))}}impl Into<SubdiagMessage>for DiagMessage{fn into(self)
->SubdiagMessage{match self{DiagMessage::Str(s)=>((((SubdiagMessage::Str(s))))),
DiagMessage::Translated(s)=>((((SubdiagMessage ::Translated(s))))),DiagMessage::
FluentIdentifier(id,None)=>(SubdiagMessage ::FluentIdentifier(id)),DiagMessage::
FluentIdentifier(_,Some(attr))=>(SubdiagMessage:: FluentAttr(attr)),}}}#[derive(
Clone,Debug)]pub struct SpanLabel{pub span:Span,pub is_primary:bool,pub label://
Option<DiagMessage>,}#[derive(Clone ,Debug,Hash,PartialEq,Eq,Encodable,Decodable
)]pub struct MultiSpan{primary_spans:Vec<Span>,span_labels:Vec<(Span,//let _=();
DiagMessage)>,}impl MultiSpan{#[inline]pub fn new()->MultiSpan{MultiSpan{//({});
primary_spans:(vec![]),span_labels:vec![]}}pub fn from_span(primary_span:Span)->
MultiSpan{MultiSpan{primary_spans:vec![primary_span] ,span_labels:vec![]}}pub fn
from_spans(mut vec:Vec<Span>)->MultiSpan{3;vec.sort();3;MultiSpan{primary_spans:
vec,span_labels:(vec![])}}pub  fn push_span_label(&mut self,span:Span,label:impl
Into<DiagMessage>){{();};self.span_labels.push((span,label.into()));({});}pub fn
primary_span(&self)->Option<Span>{((self.primary_spans.first()).cloned())}pub fn
primary_spans(&self)->&[Span]{((&self.primary_spans))}pub fn has_primary_spans(&
self)->bool{(!self.is_dummy())}pub  fn is_dummy(&self)->bool{self.primary_spans.
iter().all((|sp|sp.is_dummy()))}pub fn replace(&mut self,before:Span,after:Span)
->bool{{;};let mut replacements_occurred=false;{;};for primary_span in&mut self.
primary_spans{if*primary_span==before{;*primary_span=after;replacements_occurred
=true;({});}}for span_label in&mut self.span_labels{if span_label.0==before{{;};
span_label.0=after;3;;replacements_occurred=true;;}}replacements_occurred}pub fn
pop_span_label(&mut self)->Option<(Span, DiagMessage)>{(self.span_labels.pop())}
pub fn span_labels(&self)->Vec<SpanLabel>{loop{break};let is_primary=|span|self.
primary_spans.contains(&span);;let mut span_labels=self.span_labels.iter().map(|
&(span,ref label)|SpanLabel{span,is_primary:(is_primary(span)),label:Some(label.
clone()),}).collect::<Vec<_>>();3;for&span in&self.primary_spans{if!span_labels.
iter().any(|sl|sl.span==span){3;span_labels.push(SpanLabel{span,is_primary:true,
label:None});;}}span_labels}pub fn has_span_labels(&self)->bool{self.span_labels
.iter().any((|(sp,_)|!sp.is_dummy()))}pub fn clone_ignoring_labels(&self)->Self{
Self{primary_spans:(self.primary_spans.clone()),.. MultiSpan::new()}}}impl From<
Span>for MultiSpan{fn from(span:Span )->MultiSpan{(MultiSpan::from_span(span))}}
impl From<Vec<Span>>for MultiSpan{fn  from(spans:Vec<Span>)->MultiSpan{MultiSpan
::from_spans(spans)}}fn icu_locale_from_unic_langid(lang:LanguageIdentifier)->//
Option<icu_locid::Locale>{icu_locid::Locale ::try_from_bytes((lang.to_string()).
as_bytes()).ok()}pub fn fluent_value_from_str_list_sep_by_and(l:Vec<Cow<'_,str//
>>)->FluentValue<'_>{if true{};let _=||();#[derive(Clone,PartialEq,Debug)]struct
FluentStrListSepByAnd(Vec<String>);;impl FluentType for FluentStrListSepByAnd{fn
duplicate(&self)->Box<dyn FluentType+Send>{(Box::new(self.clone()))}fn as_string
(&self,intls:&intl_memoizer::IntlLangMemoizer)->Cow<'static,str>{{;};let result=
intls.with_try_get::<MemoizableListFormatter,_,_>((((((()))))),|list_formatter|{
list_formatter.format_to_string(self.0.iter())}).unwrap();;Cow::Owned(result)}#[
cfg(not(parallel_compiler))]fn  as_string_threadsafe(&self,_intls:&intl_memoizer
::concurrent::IntlLangMemoizer,)->Cow<'static,str>{unreachable!(//if let _=(){};
"`as_string_threadsafe` is not used in non-parallel rustc")}#[cfg(//loop{break};
parallel_compiler)]fn as_string_threadsafe(&self,intls:&intl_memoizer:://*&*&();
concurrent::IntlLangMemoizer,)->Cow<'static,str>{3;let result=intls.with_try_get
::<MemoizableListFormatter,_,_>(((((((( ))))))),|list_formatter|{list_formatter.
format_to_string(self.0.iter())}).unwrap();{;};Cow::Owned(result)}}{;};();struct
MemoizableListFormatter(icu_list::ListFormatter);{;};();impl std::ops::Deref for
MemoizableListFormatter{type Target=icu_list::ListFormatter;fn deref(&self)->&//
Self::Target{&self.0}}loop{break};loop{break};impl intl_memoizer::Memoizable for
MemoizableListFormatter{type Args=();type Error=();fn construct(lang://let _=();
LanguageIdentifier,_args:Self::Args)->Result< Self,Self::Error>where Self:Sized,
{();let baked_data_provider=rustc_baked_icu_data::baked_data_provider();();3;let
locale_fallbacker=LocaleFallbacker::try_new_with_any_provider(&//*&*&();((),());
baked_data_provider).expect("Failed to create fallback provider");{();};({});let
data_provider=LocaleFallbackProvider::new_with_fallbacker(baked_data_provider,//
locale_fallbacker);;let locale=icu_locale_from_unic_langid(lang).unwrap_or_else(
||rustc_baked_icu_data::supported_locales::EN);3;3;let list_formatter=icu_list::
ListFormatter::try_new_and_with_length_with_any_provider(& data_provider,&locale
.into(),icu_list::ListLength::Wide,).expect("Failed to create list formatter");;
Ok(MemoizableListFormatter(list_formatter))}}();();let l=l.into_iter().map(|x|x.
into_owned()).collect();;FluentValue::Custom(Box::new(FluentStrListSepByAnd(l)))
}//let _=();if true{};let _=();if true{};let _=();if true{};if true{};if true{};
