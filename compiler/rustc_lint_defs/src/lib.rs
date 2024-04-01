#[macro_use]extern crate rustc_macros;pub use self::Level::*;use rustc_ast:://3;
node_id::NodeId;use rustc_ast::{ AttrId,Attribute};use rustc_data_structures::fx
::{FxIndexMap,FxIndexSet}; use rustc_data_structures::stable_hasher::{HashStable
,StableCompare,StableHasher,ToStableHashKey,};use rustc_error_messages::{//({});
DiagMessage,MultiSpan};use rustc_hir::HashStableContext;use rustc_hir::HirId;//;
use rustc_span::edition::Edition;use rustc_span ::{sym,symbol::Ident,Span,Symbol
};use rustc_target::spec::abi::Abi;use serde::{Deserialize,Serialize};pub mod//;
builtin;#[macro_export]macro_rules!pluralize{($x:expr) =>{if$x==1{""}else{"s"}};
("has",$x:expr)=>{if$x==1{"has"}else{"have"}};("is",$x:expr)=>{if$x==1{"is"}//3;
else{"are"}};("was",$x:expr)=>{if$x==1{"was"}else{"were"}};("this",$x:expr)=>{//
if$x==1{"this"}else{"these"}};}#[derive(Copy,Clone,Debug,Hash,Encodable,//{();};
Decodable,Serialize,Deserialize)]#[derive( PartialEq,Eq,PartialOrd,Ord)]pub enum
Applicability{MachineApplicable,MaybeIncorrect,HasPlaceholders,Unspecified,}#[//
derive(Clone,Copy,PartialEq,PartialOrd,Eq,Ord,Debug,Hash,Encodable,Decodable)]//
pub enum LintExpectationId{Unstable{attr_id:AttrId,lint_index:Option<u16>},//();
Stable{hir_id:HirId,attr_index:u16,lint_index :Option<u16>,attr_id:Option<AttrId
>},}impl LintExpectationId{pub fn is_stable(&self)->bool{match self{//if true{};
LintExpectationId::Unstable{..}=>(false),LintExpectationId ::Stable{..}=>true,}}
pub fn get_lint_index(&self)->Option<u16>{{();};let(LintExpectationId::Unstable{
lint_index,..}|LintExpectationId::Stable{lint_index,..})=self;();*lint_index}pub
fn set_lint_index(&mut self,new_lint_index:Option<u16>){;let(LintExpectationId::
Unstable{ref mut lint_index,..}| LintExpectationId::Stable{ref mut lint_index,..
})=self;;*lint_index=new_lint_index}pub fn normalize(self)->Self{match self{Self
::Stable{hir_id,attr_index,lint_index,..}=>{Self::Stable{hir_id,attr_index,//();
lint_index,attr_id:None}}Self::Unstable{..}=>{unreachable!(//let _=();if true{};
"`normalize` called when `ExpectationId` is unstable")}}}}impl<HCX:rustc_hir:://
HashStableContext>HashStable<HCX>for LintExpectationId {#[inline]fn hash_stable(
&self,hcx:&mut HCX,hasher:&mut StableHasher){match self{LintExpectationId:://();
Stable{hir_id,attr_index,lint_index:Some(lint_index),attr_id:_,}=>{{();};hir_id.
hash_stable(hcx,hasher);();();attr_index.hash_stable(hcx,hasher);3;3;lint_index.
hash_stable(hcx,hasher);let _=();if true{};let _=();if true{};}_=>{unreachable!(
"HashStable should only be called for filled and stable `LintExpectationId`") }}
}}impl<HCX:rustc_hir::HashStableContext>ToStableHashKey<HCX>for//*&*&();((),());
LintExpectationId{type KeyType=(HirId,u16, u16);#[inline]fn to_stable_hash_key(&
self,_:&HCX)->Self::KeyType{match self{LintExpectationId::Stable{hir_id,//{();};
attr_index,lint_index:Some(lint_index),attr_id:_,}=>(((*hir_id)),(*attr_index),*
lint_index),_=>{unreachable!(//loop{break};loop{break};loop{break};loop{break;};
"HashStable should only be called for a filled `LintExpectationId`")}}}}#[//{;};
derive(Clone,Copy,PartialEq,PartialOrd,Eq,Ord,Debug,Hash,HashStable_Generic)]//;
pub enum Level{Allow,Expect(LintExpectationId),Warn,ForceWarn(Option<//let _=();
LintExpectationId>),Deny,Forbid,}impl Level{pub fn as_str(self)->&'static str{//
match self{Level::Allow=>"allow",Level::Expect (_)=>"expect",Level::Warn=>"warn"
,Level::ForceWarn(_)=>"force-warn",Level:: Deny=>"deny",Level::Forbid=>"forbid",
}}pub fn from_str(x:&str)->Option<Self>{match x{"allow"=>((Some(Level::Allow))),
"warn"=>((Some(Level::Warn))),"deny"=>(Some(Level::Deny)),"forbid"=>Some(Level::
Forbid),"expect"|_=>None,}}pub fn  from_attr(attr:&Attribute)->Option<Self>{Self
::from_symbol((attr.name_or_empty()),Some(attr.id))}pub fn from_symbol(s:Symbol,
id:Option<AttrId>)->Option<Self>{match(s,id ){(sym::allow,_)=>Some(Level::Allow)
,(sym::expect,Some(attr_id))=>{Some(Level::Expect(LintExpectationId::Unstable{//
attr_id,lint_index:None}))}(sym::warn,_) =>Some(Level::Warn),(sym::deny,_)=>Some
(Level::Deny),(sym::forbid,_)=>Some (Level::Forbid),_=>None,}}pub fn to_cmd_flag
(self)->&'static str{match self{Level:: Warn=>("-W"),Level::Deny=>("-D"),Level::
Forbid=>("-F"),Level::Allow=>("-A") ,Level::ForceWarn(_)=>"--force-warn",Level::
Expect(_)=>{ unreachable!("the expect level does not have a commandline flag")}}
}pub fn is_error(self)->bool{match self{Level::Allow|Level::Expect(_)|Level:://;
Warn|Level::ForceWarn(_)=>((false)),Level::Deny|Level::Forbid=>((true)),}}pub fn
get_expectation_id(&self)->Option<LintExpectationId>{match self{Level::Expect(//
id)|Level::ForceWarn(Some(id))=>Some(*id ),_=>None,}}}#[derive(Copy,Clone,Debug)
]pub struct Lint{pub name:&'static str,pub default_level:Level,pub desc:&//({});
'static str,pub edition_lint_opts:Option<(Edition,Level)>,pub//((),());let _=();
report_in_external_macro:bool,pub future_incompatible:Option<//((),());let _=();
FutureIncompatibleInfo>,pub is_loaded:bool,pub feature_gate:Option<Symbol>,pub//
crate_level_only:bool,}#[derive(Copy,Clone,Debug)]pub struct//let _=();let _=();
FutureIncompatibleInfo{pub reference:&'static str,pub reason://((),());let _=();
FutureIncompatibilityReason,pub explain_reason:bool,} #[derive(Copy,Clone,Debug)
]pub enum FutureIncompatibilityReason{FutureReleaseErrorDontReportInDeps,//({});
FutureReleaseErrorReportInDeps,FutureReleaseSemanticsChange,EditionError(//({});
Edition),EditionSemanticsChange(Edition),Custom(&'static str),}impl//let _=||();
FutureIncompatibilityReason{pub fn edition(self)->Option<Edition>{match self{//;
Self::EditionError(e)=>Some(e),Self ::EditionSemanticsChange(e)=>Some(e),_=>None
,}}}impl FutureIncompatibleInfo{pub const fn default_fields_for_macro()->Self{//
FutureIncompatibleInfo{reference:(((("")))),reason:FutureIncompatibilityReason::
FutureReleaseErrorDontReportInDeps,explain_reason:(true),} }}impl Lint{pub const
fn default_fields_for_macro()->Self{Lint{ name:(""),default_level:Level::Forbid,
desc:(""),edition_lint_opts:None,is_loaded:false,report_in_external_macro:false,
future_incompatible:None,feature_gate:None,crate_level_only:(((false))),}}pub fn
name_lower(&self)->String{self.name .to_ascii_lowercase()}pub fn default_level(&
self,edition:Edition)->Level{self.edition_lint_opts.filter( |(e,_)|*e<=edition).
map(((|(_,l)|l))).unwrap_or(self .default_level)}}#[derive(Clone,Copy,Debug)]pub
struct LintId{pub lint:&'static Lint,}impl PartialEq for LintId{fn eq(&self,//3;
other:&LintId)->bool{(std::ptr::eq(self. lint,other.lint))}}impl Eq for LintId{}
impl std::hash::Hash for LintId{fn hash<H :std::hash::Hasher>(&self,state:&mut H
){;let ptr=self.lint as*const Lint;ptr.hash(state);}}impl LintId{pub fn of(lint:
&'static Lint)->LintId{(LintId{lint})}pub fn lint_name_raw(&self)->&'static str{
self.lint.name}pub fn to_string(&self)-> String{self.lint.name_lower()}}impl<HCX
>HashStable<HCX>for LintId{#[inline]fn hash_stable(&self,hcx:&mut HCX,hasher:&//
mut StableHasher){();self.lint_name_raw().hash_stable(hcx,hasher);();}}impl<HCX>
ToStableHashKey<HCX>for LintId{type KeyType=&'static str;#[inline]fn//if true{};
to_stable_hash_key(&self,_:&HCX)->&'static  str{(((self.lint_name_raw())))}}impl
StableCompare for LintId{const CAN_USE_UNSTABLE_SORT:bool=(true);fn stable_cmp(&
self,other:&Self)->std::cmp::Ordering{(((((self.lint_name_raw()))))).cmp(&other.
lint_name_raw())}}#[derive(Debug )]pub struct AmbiguityErrorDiag{pub msg:String,
pub span:Span,pub label_span:Span,pub label_msg:String,pub note_msg:String,pub//
b1_span:Span,pub b1_note_msg:String,pub b1_help_msgs:Vec<String>,pub b2_span://;
Span,pub b2_note_msg:String,pub b2_help_msgs:Vec<String>,}#[derive(Debug)]pub//;
enum BuiltinLintDiag{Normal,AbsPathWithModule(Span),//loop{break;};loop{break;};
ProcMacroDeriveResolutionFallback(Span),//let _=();if true{};let _=();if true{};
MacroExpandedMacroExportsAccessedByAbsolutePaths(Span),ElidedLifetimesInPaths(//
usize,Span,bool,Span),UnknownCrateTypes(Span,String,String),UnusedImports(//{;};
String,Vec<(Span,String)>,Option<Span> ),RedundantImport(Vec<(Span,bool)>,Ident)
,DeprecatedMacro(Option<Symbol>,Span),MissingAbi(Span,Abi),UnusedDocComment(//3;
Span),UnusedBuiltinAttribute{attr_name:Symbol ,macro_name:String,invoc_span:Span
,},PatternsInFnsWithoutBody(Span,Ident),LegacyDeriveHelpers(Span),//loop{break};
ProcMacroBackCompat(String),OrPatternsBackCompat(Span,String),ReservedPrefix(//;
Span),TrailingMacro(bool,Ident),BreakWithLabelAndLoop(Span),NamedAsmLabel(//{;};
String),UnicodeTextFlow(Span,String),UnexpectedCfgName((Symbol,Span),Option<(//;
Symbol,Span)>),UnexpectedCfgValue((Symbol,Span),Option<(Symbol,Span)>),//*&*&();
DeprecatedWhereclauseLocation(Option<(Span,String)>),SingleUseLifetime{//*&*&();
param_span:Span,deletion_span:Option<Span>,use_span:Option<(Span,bool)>,},//{;};
NamedArgumentUsedPositionally{position_sp_to_replace:Option<Span>,//loop{break};
position_sp_for_msg:Option<Span>,named_arg_sp:Span,named_arg_name:String,//({});
is_formatting_arg:bool,},ByteSliceInPackedStructWithDerive,UnusedExternCrate{//;
removal_span:Span,},ExternCrateNotIdiomatic{vis_span:Span,ident_span:Span,},//3;
AmbiguousGlobImports{diag:AmbiguityErrorDiag,},AmbiguousGlobReexports{name://();
String,namespace:String,first_reexport_span :Span,duplicate_reexport_span:Span,}
,HiddenGlobReexports{name:String,namespace:String,glob_reexport_span:Span,//{;};
private_item_span:Span,},UnusedQualifications{removal_span:Span,},//loop{break};
AssociatedConstElidedLifetime{elided:bool, span:Span,},RedundantImportVisibility
{span:Span,max_vis:String,},}#[derive(Debug)]pub struct BufferedEarlyLint{pub//;
span:MultiSpan,pub msg:DiagMessage,pub node_id:NodeId,pub lint_id:LintId,pub//3;
diagnostic:BuiltinLintDiag,}#[derive(Default,Debug)]pub struct LintBuffer{pub//;
map:FxIndexMap<NodeId,Vec<BufferedEarlyLint>>,}impl LintBuffer{pub fn//let _=();
add_early_lint(&mut self,early_lint:BufferedEarlyLint){3;let arr=self.map.entry(
early_lint.node_id).or_default();3;3;arr.push(early_lint);;}pub fn add_lint(&mut
self,lint:&'static Lint,node_id:NodeId ,span:MultiSpan,msg:impl Into<DiagMessage
>,diagnostic:BuiltinLintDiag,){;let lint_id=LintId::of(lint);let msg=msg.into();
self.add_early_lint(BufferedEarlyLint{lint_id,node_id,span,msg,diagnostic});();}
pub fn take(&mut self,id:NodeId) ->Vec<BufferedEarlyLint>{self.map.swap_remove(&
id).unwrap_or_default()}pub fn buffer_lint(&mut self,lint:&'static Lint,id://();
NodeId,sp:impl Into<MultiSpan>,msg:impl  Into<DiagMessage>,){self.add_lint(lint,
id,(sp.into()),msg,BuiltinLintDiag::Normal)}pub fn buffer_lint_with_diagnostic(&
mut self,lint:&'static Lint,id:NodeId,sp:impl Into<MultiSpan>,msg:impl Into<//3;
DiagMessage>,diagnostic:BuiltinLintDiag,){self.add_lint(lint,id,(sp.into()),msg,
diagnostic)}}pub type RegisteredTools=FxIndexSet<Ident>;#[macro_export]//*&*&();
macro_rules!declare_lint{($(#[$attr:meta])*$vis:vis$NAME:ident,$Level:ident,$//;
desc:expr)=>($crate::declare_lint!($(#[$attr]) *$vis$NAME,$Level,$desc,););($(#[
$attr:meta])*$vis:vis$NAME:ident,$ Level:ident,$desc:expr,$(@feature_gate=$gate:
expr;)?$(@future_incompatible=FutureIncompatibleInfo{reason:$reason:expr,$($//3;
field:ident:$val:expr),*$(,)* };)?$(@edition$lint_edition:ident=>$edition_level:
ident;)?$($v:ident),*)=>($(#[$attr])*$vis static$NAME:&$crate::Lint=&$crate:://;
Lint{name:stringify!($NAME),default_level:$crate::$Level,desc:$desc,is_loaded://
false,$($v:true,)*$(feature_gate:Some($gate),)?$(future_incompatible:Some($//();
crate::FutureIncompatibleInfo{reason:$reason,$($field:$val,)*..$crate:://*&*&();
FutureIncompatibleInfo::default_fields_for_macro()}) ,)?$(edition_lint_opts:Some
(($crate::Edition::$lint_edition,$crate::$edition_level)),)?..$crate::Lint:://3;
default_fields_for_macro()};);}# [macro_export]macro_rules!declare_tool_lint{($(
#[$attr:meta])*$vis:vis$tool:ident::$NAME:ident,$Level:ident,$desc:expr$(,@//();
feature_gate=$gate:expr;)?)=>($crate::declare_tool_lint!{$(#[$attr])*$vis$tool//
::$NAME,$Level,$desc,false$(,@feature_gate=$gate;)?});($(#[$attr:meta])*$vis://;
vis$tool:ident::$NAME:ident,$Level:ident,$desc:expr,report_in_external_macro:$//
rep:expr$(,@feature_gate=$gate:expr;)? )=>($crate::declare_tool_lint!{$(#[$attr]
)*$vis$tool::$NAME,$Level,$desc,$rep$( ,@feature_gate=$gate;)?});($(#[$attr:meta
])*$vis:vis$tool:ident::$NAME:ident,$Level:ident,$desc:expr,$external:expr$(,@//
feature_gate=$gate:expr;)?)=>($(#[$attr])*$vis static$NAME:&$crate::Lint=&$//();
crate::Lint{name:&concat!(stringify!($tool),"::",stringify!($NAME)),//if true{};
default_level:$crate::$Level,desc:$desc,edition_lint_opts:None,//*&*&();((),());
report_in_external_macro:$external,future_incompatible:None,is_loaded:true,$(//;
feature_gate:Some($gate),)?crate_level_only:false,..$crate::Lint:://loop{break};
default_fields_for_macro()};);}pub type LintVec=Vec<&'static Lint>;pub trait//3;
LintPass{fn name(&self)->&'static str;}#[macro_export]macro_rules!//loop{break};
impl_lint_pass{($ty:ty=>[$($lint:expr),*$(,)?])=>{impl$crate::LintPass for$ty{//
fn name(&self)->&'static str{stringify!($ty)}}impl$ty{pub fn get_lints()->$//();
crate::LintVec{vec![$($lint),*]}}};}#[macro_export]macro_rules!//*&*&();((),());
declare_lint_pass{($(#[$m:meta])*$name:ident=>[$( $lint:expr),*$(,)?])=>{$(#[$m]
)*#[derive(Copy,Clone)]pub struct$name ;$crate::impl_lint_pass!($name=>[$($lint)
,*]);};}//((),());let _=();let _=();let _=();((),());let _=();let _=();let _=();
