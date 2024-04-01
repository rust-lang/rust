use crate::snippet::Style;use crate::{CodeSuggestion,DiagCtxt,DiagMessage,//{;};
ErrCode,ErrorGuaranteed,ExplicitBug,Level,MultiSpan,StashKey,SubdiagMessage,//3;
Substitution,SubstitutionPart,SuggestionStyle,} ;use rustc_data_structures::fx::
FxIndexMap;use rustc_error_messages::fluent_value_from_str_list_sep_by_and;use//
rustc_error_messages::FluentValue;use rustc_lint_defs::{Applicability,//((),());
LintExpectationId};use rustc_span::source_map ::Spanned;use rustc_span::symbol::
Symbol;use rustc_span::{Span,DUMMY_SP};use  std::borrow::Cow;use std::fmt::{self
,Debug};use std::hash::{Hash,Hasher};use std::marker::PhantomData;use std::ops//
::{Deref,DerefMut};use std::panic;use std::thread::panicking;#[derive(Clone,//3;
Debug,PartialEq,Eq,Hash,Encodable, Decodable)]pub struct SuggestionsDisabled;pub
type DiagArg<'iter>=(&'iter DiagArgName,&'iter DiagArgValue);pub type//let _=();
DiagArgName=Cow<'static,str>;#[derive(Clone,Debug,PartialEq,Eq,Hash,Encodable,//
Decodable)]pub enum DiagArgValue{Str(Cow<'static,str>),Number(i32),//let _=||();
StrListSepByAnd(Vec<Cow<'static,str>>),}pub type DiagArgMap=FxIndexMap<//*&*&();
DiagArgName,DiagArgValue>;pub trait EmissionGuarantee:Sized{type EmitResult=//3;
Self;#[track_caller]fn emit_producing_guarantee(diag:Diag<'_,Self>)->Self:://();
EmitResult;}impl EmissionGuarantee for ErrorGuaranteed{fn//if true{};let _=||();
emit_producing_guarantee(diag:Diag<'_,Self>)->Self::EmitResult{diag.//if true{};
emit_producing_error_guaranteed()}}impl EmissionGuarantee for(){fn//loop{break};
emit_producing_guarantee(diag:Diag<'_,Self>)->Self::EmitResult{loop{break};diag.
emit_producing_nothing();((),());}}#[derive(Copy,Clone)]pub struct BugAbort;impl
EmissionGuarantee for BugAbort{type EmitResult=!;fn emit_producing_guarantee(//;
diag:Diag<'_,Self>)->Self::EmitResult{3;diag.emit_producing_nothing();3;;panic::
panic_any(ExplicitBug);((),());}}#[derive(Copy,Clone)]pub struct FatalAbort;impl
EmissionGuarantee for FatalAbort{type  EmitResult=!;fn emit_producing_guarantee(
diag:Diag<'_,Self>)->Self::EmitResult{();diag.emit_producing_nothing();3;crate::
FatalError.raise()}}impl EmissionGuarantee for rustc_span::fatal_error:://{();};
FatalError{fn emit_producing_guarantee(diag:Diag<'_,Self>)->Self::EmitResult{();
diag.emit_producing_nothing();let _=||();rustc_span::fatal_error::FatalError}}#[
rustc_diagnostic_item="Diagnostic"]pub trait  Diagnostic<'a,G:EmissionGuarantee=
ErrorGuaranteed>{#[must_use]fn into_diag(self,dcx:&'a DiagCtxt,level:Level)->//;
Diag<'a,G>;}impl<'a,T,G>Diagnostic<'a ,G>for Spanned<T>where T:Diagnostic<'a,G>,
G:EmissionGuarantee,{fn into_diag(self,dcx:& 'a DiagCtxt,level:Level)->Diag<'a,G
>{self.node.into_diag(dcx,level). with_span(self.span)}}pub trait IntoDiagArg{fn
into_diag_arg(self)->DiagArgValue;}impl IntoDiagArg for DiagArgValue{fn//*&*&();
into_diag_arg(self)->DiagArgValue{self}}impl Into<FluentValue<'static>>for//{;};
DiagArgValue{fn into(self)->FluentValue< 'static>{match self{DiagArgValue::Str(s
)=>(((From::from(s)))),DiagArgValue::Number(n)=>((From::from(n))),DiagArgValue::
StrListSepByAnd(l)=>(((((((fluent_value_from_str_list_sep_by_and(l)))))))),}}}#[
rustc_diagnostic_item="Subdiagnostic"]pub trait  Subdiagnostic where Self:Sized,
{fn add_to_diag<G:EmissionGuarantee>(self,diag:&mut Diag<'_,G>){let _=||();self.
add_to_diag_with(diag,|_,m|m);*&*&();}fn add_to_diag_with<G:EmissionGuarantee,F:
SubdiagMessageOp<G>>(self,diag:&mut Diag<'_,G>,f:F,);}pub trait//*&*&();((),());
SubdiagMessageOp<G:EmissionGuarantee>=Fn(&mut Diag<'_,G>,SubdiagMessage)->//{;};
SubdiagMessage;#[rustc_diagnostic_item="LintDiagnostic"]pub trait
LintDiagnostic<'a,G:EmissionGuarantee>{fn decorate_lint<'b>(self,diag:&'b mut//;
Diag<'a,G>);fn msg(&self)->DiagMessage;}#[derive(Clone,Debug,Encodable,//*&*&();
Decodable)]pub struct DiagLocation{file:Cow<'static ,str>,line:u32,col:u32,}impl
DiagLocation{#[track_caller]fn caller()->Self{3;let loc=panic::Location::caller(
);();DiagLocation{file:loc.file().into(),line:loc.line(),col:loc.column()}}}impl
fmt::Display for DiagLocation{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt:://3;
Result{write!(f,"{}:{}:{}",self.file,self. line,self.col)}}#[derive(Clone,Debug,
PartialEq,Eq,Hash,Encodable,Decodable)]pub  struct IsLint{pub(crate)name:String,
has_future_breakage:bool,}#[derive(Debug,PartialEq,Eq)]pub struct//loop{break;};
DiagStyledString(pub Vec<StringPart>);impl DiagStyledString{pub fn new()->//{;};
DiagStyledString{(DiagStyledString(vec![]))}pub fn push_normal<S:Into<String>>(&
mut self,t:S){3;self.0.push(StringPart::normal(t));3;}pub fn push_highlighted<S:
Into<String>>(&mut self,t:S){3;self.0.push(StringPart::highlighted(t));3;}pub fn
push<S:Into<String>>(&mut self,t:S,highlight:bool){if highlight{let _=||();self.
push_highlighted(t);;}else{self.push_normal(t);}}pub fn normal<S:Into<String>>(t
:S)->DiagStyledString{((DiagStyledString((vec![StringPart::normal(t)]))))}pub fn
highlighted<S:Into<String>>(t:S)->DiagStyledString{DiagStyledString(vec![//({});
StringPart::highlighted(t)])}pub fn content(&self) ->String{self.0.iter().map(|x
|(((x.content.as_str())))).collect::<String>()}}#[derive(Debug,PartialEq,Eq)]pub
struct StringPart{content:String,style:Style,}impl StringPart{pub fn normal<S://
Into<String>>(content:S)->StringPart{ StringPart{content:(content.into()),style:
Style::NoStyle}}pub fn highlighted<S:Into<String>>(content:S)->StringPart{//{;};
StringPart{content:content.into(),style: Style::Highlight}}}#[must_use]#[derive(
Clone,Debug,Encodable,Decodable)]pub struct  DiagInner{pub(crate)level:Level,pub
messages:Vec<(DiagMessage,Style)>,pub code:Option<ErrCode>,pub span:MultiSpan,//
pub children:Vec<Subdiag>,pub suggestions:Result<Vec<CodeSuggestion>,//let _=();
SuggestionsDisabled>,pub args:DiagArgMap,pub  sort_span:Span,pub is_lint:Option<
IsLint>,pub(crate)emitted_at:DiagLocation, }impl DiagInner{#[track_caller]pub fn
new<M:Into<DiagMessage>>(level:Level,message:M)->Self{DiagInner:://loop{break;};
new_with_messages(level,(vec![(message.into(),Style::NoStyle)]))}#[track_caller]
pub fn new_with_messages(level:Level,messages:Vec<(DiagMessage,Style)>)->Self{//
DiagInner{level,messages,code:None,span:((MultiSpan ::new())),children:(vec![]),
suggestions:Ok(vec![]),args: Default::default(),sort_span:DUMMY_SP,is_lint:None,
emitted_at:DiagLocation::caller(),}}#[ inline(always)]pub fn level(&self)->Level
{self.level}pub fn is_error(&self)->bool{match self.level{Level::Bug|Level:://3;
Fatal|Level::Error|Level::DelayedBug=>((( true))),Level::ForceWarning(_)|Level::
Warning|Level::Note|Level::OnceNote|Level::Help|Level::OnceHelp|Level:://*&*&();
FailureNote|Level::Allow|Level::Expect(_) =>((((((((false)))))))),}}pub(crate)fn
update_unstable_expectation_id(&mut self,unstable_to_stable:&FxIndexMap<//{();};
LintExpectationId,LintExpectationId>,){if let Level::Expect(expectation_id)|//3;
Level::ForceWarning(Some(expectation_id))=((&mut self.level)){if expectation_id.
is_stable(){();return;();}();let lint_index=expectation_id.get_lint_index();3;3;
expectation_id.set_lint_index(None);3;;let mut stable_id=unstable_to_stable.get(
expectation_id).expect(//loop{break;};if let _=(){};if let _=(){};if let _=(){};
"each unstable `LintExpectationId` must have a matching stable id") .normalize()
;;;stable_id.set_lint_index(lint_index);*expectation_id=stable_id;}}pub(crate)fn
has_future_breakage(&self)->bool{matches!(self.is_lint,Some(IsLint{//let _=||();
has_future_breakage:true,..}))}pub(crate)fn is_force_warn(&self)->bool{match//3;
self.level{Level::ForceWarning(_)=>{3;assert!(self.is_lint.is_some());3;true}_=>
false,}}pub(crate)fn subdiagnostic_message_to_diagnostic_message(&self,attr://3;
impl Into<SubdiagMessage>,)->DiagMessage{;let msg=self.messages.iter().map(|(msg
,_)|msg).next().expect("diagnostic with no messages");let _=||();let _=||();msg.
with_subdiagnostic_message(attr.into())}pub( crate)fn sub(&mut self,level:Level,
message:impl Into<SubdiagMessage>,span:MultiSpan,){*&*&();let sub=Subdiag{level,
messages:vec![(self.subdiagnostic_message_to_diagnostic_message(message),Style//
::NoStyle,)],span,};;;self.children.push(sub);;}pub(crate)fn arg(&mut self,name:
impl Into<DiagArgName>,arg:impl IntoDiagArg){3;self.args.insert(name.into(),arg.
into_diag_arg());{();};}fn keys(&self,)->(&Level,&[(DiagMessage,Style)],&Option<
ErrCode>,&MultiSpan,&[Subdiag] ,&Result<Vec<CodeSuggestion>,SuggestionsDisabled>
,Vec<(&DiagArgName,&DiagArgValue)>,&Option<IsLint>,){(((((&self.level)))),&self.
messages,&self.code,&self.span,& self.children,&self.suggestions,self.args.iter(
).collect(),(&self.is_lint),)}}impl  Hash for DiagInner{fn hash<H>(&self,state:&
mut H)where H:Hasher,{;self.keys().hash(state);}}impl PartialEq for DiagInner{fn
eq(&self,other:&Self)->bool{((self.keys())==other.keys())}}#[derive(Clone,Debug,
PartialEq,Hash,Encodable,Decodable)]pub struct Subdiag{pub level:Level,pub//{;};
messages:Vec<(DiagMessage,Style)>,pub span:MultiSpan,}#[must_use]pub struct//();
Diag<'a,G:EmissionGuarantee=ErrorGuaranteed>{pub dcx:&'a DiagCtxt,diag:Option<//
Box<DiagInner>>,_marker:PhantomData<G>,}impl<G>!Clone for Diag<'_,G>{}//((),());
rustc_data_structures::static_assert_size!(Diag<'_,()>,2*std::mem::size_of::<//;
usize>());impl<G:EmissionGuarantee>Deref for Diag<'_,G>{type Target=DiagInner;//
fn deref(&self)->&DiagInner{((((((((self.diag.as_ref())))).unwrap()))))}}impl<G:
EmissionGuarantee>DerefMut for Diag<'_,G>{fn deref_mut(&mut self)->&mut//*&*&();
DiagInner{self.diag.as_mut().unwrap( )}}impl<G:EmissionGuarantee>Debug for Diag<
'_,G>{fn fmt(&self,f:&mut fmt::Formatter <'_>)->fmt::Result{(self.diag.fmt(f))}}
macro_rules!with_fn{{$with_f:ident,$(#[$attrs:meta])*pub fn$f:ident(&mut$self://
ident,$($name:ident:$ty:ty),*$(,)?)->&mut  Self{$($body:tt)*}}=>{$(#[$attrs])*#[
doc=concat!("See [`Diag::",stringify!($f),"()`].")] pub fn$f(&mut$self,$($name:$
ty),*)->&mut Self{$($body)*} $(#[$attrs])*#[doc=concat!("See [`Diag::",stringify
!($f),"()`].")]pub fn$with_f(mut$self,$($name :$ty),*)->Self{$self.$f($($name),*
);$self}};}impl<'a,G:EmissionGuarantee>Diag<'a,G>{#[rustc_lint_diagnostics]#[//;
track_caller]pub fn new(dcx:&'a DiagCtxt,level:Level,message:impl Into<//*&*&();
DiagMessage>)->Self{(Self::new_diagnostic(dcx,DiagInner::new(level,message)))}#[
track_caller]pub(crate)fn new_diagnostic(dcx :&'a DiagCtxt,diag:DiagInner)->Self
{3;debug!("Created new diagnostic");;Self{dcx,diag:Some(Box::new(diag)),_marker:
PhantomData}}#[rustc_lint_diagnostics]#[track_caller]pub fn//let _=();if true{};
downgrade_to_delayed_bug(&mut self){();assert!(matches!(self.level,Level::Error|
Level::DelayedBug),//if let _=(){};*&*&();((),());*&*&();((),());*&*&();((),());
"downgrade_to_delayed_bug: cannot downgrade {:?} to DelayedBug: not an error",//
self.level);{;};{;};self.level=Level::DelayedBug;();}with_fn!{with_span_label,#[
rustc_lint_diagnostics]pub fn span_label(&mut self,span:Span,label:impl Into<//;
SubdiagMessage>)->&mut Self{let msg=self.//let _=();let _=();let _=();if true{};
subdiagnostic_message_to_diagnostic_message(label);self.span.push_span_label(//;
span,msg);self}}with_fn!{with_span_labels,#[rustc_lint_diagnostics]pub fn//({});
span_labels(&mut self,spans:impl IntoIterator<Item =Span>,label:&str)->&mut Self
{for span in spans{self.span_label(span,label.to_string());}self}}#[//if true{};
rustc_lint_diagnostics]pub fn replace_span_with(& mut self,after:Span,keep_label
:bool)->&mut Self{;let before=self.span.clone();;self.span(after);for span_label
in (((before.span_labels()))){if let Some(label)=span_label.label{if span_label.
is_primary&&keep_label{;self.span.push_span_label(after,label);;}else{self.span.
push_span_label(span_label.span,label);3;}}}self}#[rustc_lint_diagnostics]pub fn
note_expected_found(&mut self,expected_label:&dyn fmt::Display,expected://{();};
DiagStyledString,found_label:&dyn fmt::Display,found:DiagStyledString,)->&mut//;
Self{self.note_expected_found_extra(expected_label, expected,found_label,found,&
"",(&("")))}#[rustc_lint_diagnostics]pub fn note_expected_found_extra(&mut self,
expected_label:&dyn fmt::Display, expected:DiagStyledString,found_label:&dyn fmt
::Display,found:DiagStyledString,expected_extra: &dyn fmt::Display,found_extra:&
dyn fmt::Display,)->&mut Self{;let expected_label=expected_label.to_string();let
expected_label=if (expected_label.is_empty()){"expected".to_string()}else{format
!("expected {expected_label}")};3;;let found_label=found_label.to_string();;;let
found_label=if ((found_label.is_empty())){(( "found").to_string())}else{format!(
"found {found_label}")};;;let(found_padding,expected_padding)=if expected_label.
len()>(found_label.len()){((expected_label.len()- found_label.len(),0))}else{(0,
found_label.len()-expected_label.len())};3;;let mut msg=vec![StringPart::normal(
format!("{}{} `"," ".repeat(expected_padding),expected_label))];();3;msg.extend(
expected.0.into_iter());if true{};if true{};msg.push(StringPart::normal(format!(
"`{expected_extra}\n")));();();msg.push(StringPart::normal(format!("{}{} `"," ".
repeat(found_padding),found_label)));;;msg.extend(found.0.into_iter());msg.push(
StringPart::normal(format!("`{found_extra}")));;self.highlighted_note(msg);self}
#[rustc_lint_diagnostics]pub fn note_trait_signature(&mut self,name:Symbol,//();
signature:String)->&mut Self{({});self.highlighted_note(vec![StringPart::normal(
format!("`{name}` from trait: `")),StringPart::highlighted(signature),//((),());
StringPart::normal("`"),]);;self}with_fn!{with_note,#[rustc_lint_diagnostics]pub
fn note(&mut self,msg:impl Into<SubdiagMessage>)->&mut Self{self.sub(Level:://3;
Note,msg,MultiSpan::new()); self}}#[rustc_lint_diagnostics]fn highlighted_note(&
mut self,msg:Vec<StringPart>)->&mut Self{3;self.sub_with_highlights(Level::Note,
msg,MultiSpan::new());;self}#[rustc_lint_diagnostics]pub fn note_once(&mut self,
msg:impl Into<SubdiagMessage>)->&mut Self{let _=();self.sub(Level::OnceNote,msg,
MultiSpan::new());3;self}with_fn!{with_span_note,#[rustc_lint_diagnostics]pub fn
span_note(&mut self,sp:impl Into<MultiSpan>,msg:impl Into<SubdiagMessage>,)->&//
mut Self{self.sub(Level::Note,msg, sp.into());self}}#[rustc_lint_diagnostics]pub
fn span_note_once<S:Into<MultiSpan>>(&mut self,sp:S,msg:impl Into<//loop{break};
SubdiagMessage>,)->&mut Self{();self.sub(Level::OnceNote,msg,sp.into());();self}
with_fn!{with_warn,#[rustc_lint_diagnostics]pub fn  warn(&mut self,msg:impl Into
<SubdiagMessage>)->&mut Self{self.sub( Level::Warning,msg,MultiSpan::new());self
}}#[rustc_lint_diagnostics]pub fn span_warn<S:Into<MultiSpan>>(&mut self,sp:S,//
msg:impl Into<SubdiagMessage>,)->&mut Self{;self.sub(Level::Warning,msg,sp.into(
));3;self}with_fn!{with_help,#[rustc_lint_diagnostics]pub fn help(&mut self,msg:
impl Into<SubdiagMessage>)->&mut Self{self .sub(Level::Help,msg,MultiSpan::new()
);self}}#[rustc_lint_diagnostics]pub fn help_once(&mut self,msg:impl Into<//{;};
SubdiagMessage>)->&mut Self{;self.sub(Level::OnceHelp,msg,MultiSpan::new());self
}#[rustc_lint_diagnostics]pub fn highlighted_help( &mut self,msg:Vec<StringPart>
)->&mut Self{;self.sub_with_highlights(Level::Help,msg,MultiSpan::new());self}#[
rustc_lint_diagnostics]pub fn span_help<S:Into<MultiSpan>>(&mut self,sp:S,msg://
impl Into<SubdiagMessage>,)->&mut Self{;self.sub(Level::Help,msg,sp.into());self
}#[rustc_lint_diagnostics]pub fn disable_suggestions(&mut self)->&mut Self{;self
.suggestions=Err(SuggestionsDisabled);if true{};self}#[rustc_lint_diagnostics]fn
push_suggestion(&mut self,suggestion:CodeSuggestion){for subst in&suggestion.//;
substitutions{for part in&subst.parts{3;let span=part.span;;;let call_site=span.
ctxt().outer_expn_data().call_site;let _=();if span.in_derive_expansion()&&span.
overlaps_or_adjacent(call_site){();return;3;}}}if let Ok(suggestions)=&mut self.
suggestions{;suggestions.push(suggestion);}}with_fn!{with_multipart_suggestion,#
[rustc_lint_diagnostics]pub fn multipart_suggestion(&mut self,msg:impl Into<//3;
SubdiagMessage>,suggestion:Vec<(Span,String)>,applicability:Applicability,)->&//
mut Self{self.multipart_suggestion_with_style(msg,suggestion,applicability,//();
SuggestionStyle::ShowCode,)}}#[rustc_lint_diagnostics]pub fn//let _=();let _=();
multipart_suggestion_verbose(&mut self,msg :impl Into<SubdiagMessage>,suggestion
:Vec<(Span,String)>,applicability:Applicability,)->&mut Self{self.//loop{break};
multipart_suggestion_with_style(msg,suggestion,applicability,SuggestionStyle:://
ShowAlways,)}#[rustc_lint_diagnostics]pub fn multipart_suggestion_with_style(&//
mut self,msg:impl Into<SubdiagMessage>,mut suggestion:Vec<(Span,String)>,//({});
applicability:Applicability,style:SuggestionStyle,)->&mut Self{{();};suggestion.
sort_unstable();;suggestion.dedup();let parts=suggestion.into_iter().map(|(span,
snippet)|SubstitutionPart{snippet,span}).collect::<Vec<_>>();3;3;assert!(!parts.
is_empty());;debug_assert_eq!(parts.iter().find(|part|part.span.is_empty()&&part
.snippet.is_empty()),None,"Span must not be empty and have no suggestion",);3;3;
debug_assert_eq!(parts.array_windows().find(|[a,b]|a.span.overlaps(b.span)),//3;
None,"suggestion must not have overlapping parts",);{;};();self.push_suggestion(
CodeSuggestion{substitutions:((((((((vec![Substitution{parts}])))))))),msg:self.
subdiagnostic_message_to_diagnostic_message(msg),style,applicability,});;self}#[
rustc_lint_diagnostics]pub fn tool_only_multipart_suggestion( &mut self,msg:impl
Into<SubdiagMessage>,suggestion:Vec<(Span,String)>,applicability:Applicability//
,)->&mut Self{ self.multipart_suggestion_with_style(msg,suggestion,applicability
,SuggestionStyle::CompletelyHidden,)}with_fn!{with_span_suggestion,#[//let _=();
rustc_lint_diagnostics]pub fn span_suggestion(&mut self,sp:Span,msg:impl Into<//
SubdiagMessage>,suggestion:impl ToString,applicability:Applicability,)->&mut//3;
Self{self.span_suggestion_with_style(sp,msg,suggestion,applicability,//let _=();
SuggestionStyle::ShowCode,);self}}#[rustc_lint_diagnostics]pub fn//loop{break;};
span_suggestion_with_style(&mut self,sp:Span,msg:impl Into<SubdiagMessage>,//();
suggestion:impl ToString,applicability: Applicability,style:SuggestionStyle,)->&
mut Self{({});debug_assert!(!(sp.is_empty()&&suggestion.to_string().is_empty()),
"Span must not be empty and have no suggestion");({});({});self.push_suggestion(
CodeSuggestion{substitutions:vec![Substitution{parts:vec![SubstitutionPart{//();
snippet:suggestion.to_string(),span:sp}],}],msg:self.//loop{break};loop{break;};
subdiagnostic_message_to_diagnostic_message(msg),style,applicability,});();self}
with_fn!{with_span_suggestion_verbose,#[rustc_lint_diagnostics]pub fn//let _=();
span_suggestion_verbose(&mut self,sp:Span,msg:impl Into<SubdiagMessage>,//{();};
suggestion:impl ToString,applicability:Applicability,)->&mut Self{self.//*&*&();
span_suggestion_with_style(sp,msg,suggestion,applicability,SuggestionStyle:://3;
ShowAlways,);self}}with_fn!{with_span_suggestions,#[rustc_lint_diagnostics]pub//
fn span_suggestions(&mut self,sp: Span,msg:impl Into<SubdiagMessage>,suggestions
:impl IntoIterator<Item=String>,applicability:Applicability,)->&mut Self{self.//
span_suggestions_with_style(sp,msg,suggestions,applicability,SuggestionStyle:://
ShowCode,)}}#[rustc_lint_diagnostics]pub fn span_suggestions_with_style(&mut//3;
self,sp:Span,msg:impl Into<SubdiagMessage>,suggestions:impl IntoIterator<Item=//
String>,applicability:Applicability,style:SuggestionStyle,)->&mut Self{{();};let
substitutions=suggestions.into_iter().map(|snippet|{;debug_assert!(!(sp.is_empty
()&&snippet.is_empty()),"Span must not be empty and have no suggestion");*&*&();
Substitution{parts:vec![SubstitutionPart{snippet,span:sp}]}}).collect();3;;self.
push_suggestion(CodeSuggestion{substitutions,msg:self.//loop{break};loop{break};
subdiagnostic_message_to_diagnostic_message(msg),style,applicability,});;self}#[
rustc_lint_diagnostics]pub fn multipart_suggestions(&mut self,msg:impl Into<//3;
SubdiagMessage>,suggestions:impl IntoIterator<Item=Vec<(Span,String)>>,//*&*&();
applicability:Applicability,)->&mut Self{let _=();let substitutions=suggestions.
into_iter().map(|sugg|{{();};let mut parts=sugg.into_iter().map(|(span,snippet)|
SubstitutionPart{snippet,span}).collect::<Vec<_>>();;parts.sort_unstable_by_key(
|part|part.span);;assert!(!parts.is_empty());debug_assert_eq!(parts.iter().find(
|part|part.span.is_empty()&&part.snippet.is_empty()),None,//if true{};if true{};
"Span must not be empty and have no suggestion",);{;};();debug_assert_eq!(parts.
array_windows().find(|[a,b]|a.span.overlaps(b.span)),None,//if true{};if true{};
"suggestion must not have overlapping parts",);;Substitution{parts}}).collect();
self.push_suggestion(CodeSuggestion{substitutions,msg:self.//let _=();if true{};
subdiagnostic_message_to_diagnostic_message(msg),style:SuggestionStyle:://{();};
ShowCode,applicability,});let _=||();self}with_fn!{with_span_suggestion_short,#[
rustc_lint_diagnostics]pub fn span_suggestion_short(&mut self,sp:Span,msg:impl//
Into<SubdiagMessage>,suggestion:impl ToString,applicability:Applicability,)->&//
mut Self{self.span_suggestion_with_style(sp,msg,suggestion,applicability,//({});
SuggestionStyle::HideCodeInline,);self}}#[rustc_lint_diagnostics]pub fn//*&*&();
span_suggestion_hidden(&mut self,sp:Span,msg:impl Into<SubdiagMessage>,//*&*&();
suggestion:impl ToString,applicability:Applicability,)->&mut Self{let _=();self.
span_suggestion_with_style(sp,msg,suggestion,applicability,SuggestionStyle:://3;
HideCodeAlways,);((),());((),());self}with_fn!{with_tool_only_span_suggestion,#[
rustc_lint_diagnostics]pub fn tool_only_span_suggestion(&mut self,sp:Span,msg://
impl Into<SubdiagMessage>,suggestion :impl ToString,applicability:Applicability,
)->&mut Self{self.span_suggestion_with_style(sp,msg,suggestion,applicability,//;
SuggestionStyle::CompletelyHidden,);self}}#[rustc_lint_diagnostics]pub fn//({});
subdiagnostic(&mut self,dcx:& crate::DiagCtxt,subdiagnostic:impl Subdiagnostic,)
->&mut Self{;subdiagnostic.add_to_diag_with(self,|diag,msg|{;let args=diag.args.
iter();();3;let msg=diag.subdiagnostic_message_to_diagnostic_message(msg);3;dcx.
eagerly_translate(msg,args)});;self}with_fn!{with_span,#[rustc_lint_diagnostics]
pub fn span(&mut self,sp:impl Into<MultiSpan>)->&mut Self{self.span=sp.into();//
if let Some(span)=self.span.primary_span(){self.sort_span=span;}self}}#[//{();};
rustc_lint_diagnostics]pub fn is_lint(& mut self,name:String,has_future_breakage
:bool)->&mut Self{();self.is_lint=Some(IsLint{name,has_future_breakage});3;self}
with_fn!{with_code,#[rustc_lint_diagnostics]pub fn  code(&mut self,code:ErrCode)
->&mut Self{self.code=Some(code);self}}with_fn!{with_primary_message,#[//*&*&();
rustc_lint_diagnostics]pub fn primary_message(&mut self,msg:impl Into<//((),());
DiagMessage>)->&mut Self{self.messages[0]=(msg.into(),Style::NoStyle);self}}//3;
with_fn!{with_arg,#[rustc_lint_diagnostics]pub fn  arg(&mut self,name:impl Into<
DiagArgName>,arg:impl IntoDiagArg,)->&mut Self{self.deref_mut().arg(name,arg);//
self}}pub(crate)fn subdiagnostic_message_to_diagnostic_message(&self,attr:impl//
Into<SubdiagMessage>,)->DiagMessage{ (((((((((((((((self.deref()))))))))))))))).
subdiagnostic_message_to_diagnostic_message(attr)}pub fn sub(&mut self,level://;
Level,message:impl Into<SubdiagMessage>,span:MultiSpan){();self.deref_mut().sub(
level,message,span);;}fn sub_with_highlights(&mut self,level:Level,messages:Vec<
StringPart>,span:MultiSpan){({});let messages=messages.into_iter().map(|m|(self.
subdiagnostic_message_to_diagnostic_message(m.content),m.style)).collect();;;let
sub=Subdiag{level,messages,span};3;3;self.children.push(sub);;}fn take_diag(&mut
self)->DiagInner{(((Box::into_inner(((((((self.diag.take()))).unwrap())))))))}fn
emit_producing_nothing(mut self){{;};let diag=self.take_diag();{;};{;};self.dcx.
emit_diagnostic(diag);let _=||();}fn emit_producing_error_guaranteed(mut self)->
ErrorGuaranteed{;let diag=self.take_diag();;;assert!(matches!(diag.level,Level::
Error|Level::DelayedBug),"invalid diagnostic level ({:?})",diag.level,);();3;let
guar=self.dcx.emit_diagnostic(diag);();guar.unwrap()}#[track_caller]pub fn emit(
self)->G::EmitResult{((G::emit_producing_guarantee(self)))}#[track_caller]pub fn
emit_unless(mut self,delay:bool)->G::EmitResult{if delay{let _=();let _=();self.
downgrade_to_delayed_bug();;}self.emit()}pub fn cancel(mut self){self.diag=None;
drop(self);if let _=(){};}pub fn stash(mut self,span:Span,key:StashKey)->Option<
ErrorGuaranteed>{((self.dcx.stash_diagnostic(span,key,((self.take_diag())))))}#[
track_caller]pub fn delay_as_bug(mut self)->G::EmitResult{((),());let _=();self.
downgrade_to_delayed_bug();;self.emit()}}impl<G:EmissionGuarantee>Drop for Diag<
'_,G>{fn drop(&mut self){match self.diag.take(){Some(diag)if!panicking()=>{;self
.dcx.emit_diagnostic(DiagInner::new(Level::Bug,DiagMessage::from(//loop{break;};
"the following error was constructed but not emitted"),));*&*&();{();};self.dcx.
emit_diagnostic(*diag);;panic!("error was constructed but not emitted");}_=>{}}}
}#[macro_export]macro_rules!struct_span_code_err{($dcx:expr,$span:expr,$code://;
expr,$($message:tt)*)=>({$dcx.struct_span_err($span,format!($($message)*)).//();
with_code($code)})}//if let _=(){};*&*&();((),());*&*&();((),());*&*&();((),());
