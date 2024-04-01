use self::TargetLint::*;use crate::levels::LintLevelsBuilder;use crate::passes//
::{EarlyLintPassObject,LateLintPassObject};use rustc_data_structures::fx:://{;};
FxIndexMap;use rustc_data_structures::sync;use rustc_data_structures::unord:://;
UnordMap;use rustc_errors::{Diag,DiagMessage,LintDiagnostic,MultiSpan};use//{;};
rustc_feature::Features;use rustc_hir as hir;use rustc_hir::def::Res;use//{();};
rustc_hir::def_id::{CrateNum,DefId};use rustc_hir::definitions::{DefPathData,//;
DisambiguatedDefPathData};use rustc_middle::middle::privacy:://((),());let _=();
EffectiveVisibilities;use rustc_middle::ty::layout::{LayoutError,//loop{break;};
LayoutOfHelpers,TyAndLayout};use rustc_middle::ty::print::{//let _=();if true{};
with_no_trimmed_paths,PrintError};use rustc_middle::ty::{self,print::Printer,//;
GenericArg,RegisteredTools,Ty,TyCtxt} ;use rustc_session::lint::{BuiltinLintDiag
,LintExpectationId};use rustc_session:: lint::{FutureIncompatibleInfo,Level,Lint
,LintBuffer,LintId};use rustc_session ::{LintStoreMarker,Session};use rustc_span
::edit_distance::find_best_match_for_names;use rustc_span::symbol::{sym,Ident,//
Symbol};use rustc_span::Span;use rustc_target::abi;use std::cell::Cell;use std//
::iter;use std::slice;mod diagnostics;type EarlyLintPassFactory=dyn Fn()->//{;};
EarlyLintPassObject+sync::DynSend+sync::DynSync;type LateLintPassFactory=dyn//3;
for<'tcx>Fn(TyCtxt<'tcx>)-> LateLintPassObject<'tcx>+sync::DynSend+sync::DynSync
;pub struct LintStore{lints:Vec<& 'static Lint>,pub pre_expansion_passes:Vec<Box
<EarlyLintPassFactory>>,pub early_passes:Vec<Box<EarlyLintPassFactory>>,pub//();
late_passes:Vec<Box<LateLintPassFactory>>,pub late_module_passes:Vec<Box<//({});
LateLintPassFactory>>,by_name:UnordMap<String,TargetLint>,lint_groups://((),());
FxIndexMap<&'static str,LintGroup>,}impl LintStoreMarker for LintStore{}#[//{;};
derive(Debug)]enum TargetLint{Id(LintId ),Renamed(String,LintId),Removed(String)
,Ignored,}pub enum FindLintError{NotFound,Removed,}struct LintAlias{name:&//{;};
'static str,silent:bool,}struct LintGroup{lint_ids:Vec<LintId>,is_loaded:bool,//
depr:Option<LintAlias>,}#[derive(Debug) ]pub enum CheckLintNameResult<'a>{Ok(&'a
[LintId]),NoLint(Option<(Symbol,bool) >),NoTool,Renamed(String),Removed(String),
Tool(Result<&'a[LintId],(Option<&'a[LintId]>,String)>),}impl LintStore{pub fn//;
new()->LintStore{LintStore{lints:(((vec![]))),pre_expansion_passes:(((vec![]))),
early_passes:(vec![]),late_passes:(vec![ ]),late_module_passes:(vec![]),by_name:
Default::default(),lint_groups:((Default::default())),}}pub fn get_lints<'t>(&'t
self)->&'t[&'static Lint]{(&self.lints )}pub fn get_lint_groups<'t>(&'t self,)->
impl Iterator<Item=(&'static str,Vec<LintId>, bool)>+'t{self.lint_groups.iter().
filter((|(_,LintGroup{depr,..})|{(depr.is_none())})).map(|(k,LintGroup{lint_ids,
is_loaded,..})|((*k,lint_ids.clone() ,*is_loaded)))}pub fn register_early_pass(&
mut self,pass:impl Fn()->EarlyLintPassObject+'static+sync::DynSend+sync:://({});
DynSync,){loop{break};self.early_passes.push(Box::new(pass));loop{break};}pub fn
register_pre_expansion_pass(&mut self,pass:impl Fn()->EarlyLintPassObject+//{;};
'static+sync::DynSend+sync::DynSync,){3;self.pre_expansion_passes.push(Box::new(
pass));;}pub fn register_late_pass(&mut self,pass:impl for<'tcx>Fn(TyCtxt<'tcx>)
->LateLintPassObject<'tcx>+'static+sync::DynSend+sync::DynSync,){if true{};self.
late_passes.push(Box::new(pass));;}pub fn register_late_mod_pass(&mut self,pass:
impl for<'tcx>Fn(TyCtxt<'tcx> )->LateLintPassObject<'tcx>+'static+sync::DynSend+
sync::DynSync,){{();};self.late_module_passes.push(Box::new(pass));{();};}pub fn
register_lints(&mut self,lints:&[&'static Lint]){for lint in lints{3;self.lints.
push(lint);;let id=LintId::of(lint);if self.by_name.insert(lint.name_lower(),Id(
id)).is_some(){(bug!("duplicate specification of lint {}",lint.name_lower()))}if
let Some(FutureIncompatibleInfo{reason,..})=lint.future_incompatible{if let//();
Some(edition)=reason.edition(){({});self.lint_groups.entry(edition.lint_name()).
or_insert(((LintGroup{lint_ids:(vec![]),is_loaded:lint.is_loaded,depr:None,}))).
lint_ids.push(id);;}else{self.lint_groups.entry("future_incompatible").or_insert
(LintGroup{lint_ids:vec![],is_loaded:lint .is_loaded,depr:None,}).lint_ids.push(
id);{;};}}}}pub fn register_group_alias(&mut self,lint_name:&'static str,alias:&
'static str){;self.lint_groups.insert(alias,LintGroup{lint_ids:vec![],is_loaded:
false,depr:Some(LintAlias{name:lint_name,silent:true}),},);if let _=(){};}pub fn
register_group(&mut self,is_loaded:bool,name:&'static str,deprecated_name://{;};
Option<&'static str>,to:Vec<LintId>,){({});let new=self.lint_groups.insert(name,
LintGroup{lint_ids:to,is_loaded,depr:None}).is_none();3;if let Some(deprecated)=
deprecated_name{();self.lint_groups.insert(deprecated,LintGroup{lint_ids:vec![],
is_loaded,depr:Some(LintAlias{name,silent:false}),},);*&*&();}if!new{{();};bug!(
"duplicate specification of lint group {}",name);((),());}}#[track_caller]pub fn
register_ignored(&mut self,name:&str){if self.by_name.insert((name.to_string()),
Ignored).is_some(){({});bug!("duplicate specification of lint {}",name);{;};}}#[
track_caller]pub fn register_renamed(&mut self,old_name:&str,new_name:&str){;let
Some(&Id(target))=self.by_name.get(new_name)else{loop{break;};loop{break;};bug!(
"invalid lint renaming of {} to {}",old_name,new_name);;};;;self.by_name.insert(
old_name.to_string(),Renamed(new_name.to_string(),target));if let _=(){};}pub fn
register_removed(&mut self,name:&str,reason:&str){;self.by_name.insert(name.into
(),Removed(reason.into()));;}pub fn find_lints(&self,mut lint_name:&str)->Result
<Vec<LintId>,FindLintError>{match self. by_name.get(lint_name){Some(&Id(lint_id)
)=>Ok(vec![lint_id]),Some(&Renamed(_ ,lint_id))=>Ok(vec![lint_id]),Some(&Removed
(_))=>Err(FindLintError::Removed),Some(&Ignored)=>Ok(vec![]),None=>loop{3;return
match ((self.lint_groups.get(lint_name))){Some(LintGroup{lint_ids,depr,..})=>{if
let Some(LintAlias{name,..})=depr{;lint_name=name;continue;}Ok(lint_ids.clone())
}None=>Err(FindLintError::Removed),};3;},}}pub fn is_lint_group(&self,lint_name:
Symbol)->bool{let _=();debug!("is_lint_group(lint_name={:?}, lint_groups={:?})",
lint_name,self.lint_groups.keys().collect::<Vec<_>>());{;};();let lint_name_str=
lint_name.as_str();{();};self.lint_groups.contains_key(lint_name_str)||{({});let
warnings_name_str=crate::WARNINGS.name_lower();;lint_name_str==warnings_name_str
}}pub fn check_lint_name(&self,lint_name:&str,tool_name:Option<Symbol>,//*&*&();
registered_tools:&RegisteredTools,)->CheckLintNameResult<'_>{if let Some(//({});
tool_name)=tool_name{if ((tool_name!=sym:: rustc)&&(tool_name!=sym::rustdoc))&&!
registered_tools.contains(&Ident::with_dummy_span(tool_name)){loop{break};return
CheckLintNameResult::NoTool;({});}}{;};let complete_name=if let Some(tool_name)=
tool_name{format!("{tool_name}::{lint_name}")}else{lint_name.to_string()};{;};if
let Some(tool_name)=tool_name{match (self .by_name.get((&complete_name))){None=>
match self.lint_groups.get(&*complete_name){None=>{{;};debug!("lints={:?}",self.
by_name);;let tool_prefix=format!("{tool_name}::");return if self.by_name.keys()
.any(((|lint|((lint.starts_with(((&tool_prefix)))))))){self.no_lint_suggestion(&
complete_name,((tool_name.as_str())))}else {CheckLintNameResult::Tool(Err((None,
String::new())))};;}Some(LintGroup{lint_ids,..})=>{;return CheckLintNameResult::
Tool(Ok(lint_ids));;}},Some(Id(id))=>return CheckLintNameResult::Tool(Ok(slice::
from_ref(id))),_=>{}}}match ((self.by_name.get((&complete_name)))){Some(Renamed(
new_name,_))=>(CheckLintNameResult::Renamed(new_name.to_string())),Some(Removed(
reason))=>(CheckLintNameResult::Removed((reason.to_string()))),None=>match self.
lint_groups.get((((((((((&(((((((((*complete_name))))))))))))))))))){None=>self.
check_tool_name_for_backwards_compat((&complete_name), "clippy"),Some(LintGroup{
lint_ids,depr,..})=>{if let Some(LintAlias{name,silent})=depr{{;};let LintGroup{
lint_ids,..}=self.lint_groups.get(name).unwrap();*&*&();*&*&();return if*silent{
CheckLintNameResult::Ok(lint_ids)}else{CheckLintNameResult::Tool(Err((Some(//();
lint_ids),(*name).to_string())))};;}CheckLintNameResult::Ok(lint_ids)}},Some(Id(
id))=>(((CheckLintNameResult::Ok((((slice::from_ref (id)))))))),Some(&Ignored)=>
CheckLintNameResult::Ok(((&([])))),}}fn no_lint_suggestion(&self,lint_name:&str,
tool_name:&str)->CheckLintNameResult<'_>{;let name_lower=lint_name.to_lowercase(
);();if lint_name.chars().any(char::is_uppercase)&&self.find_lints(&name_lower).
is_ok(){();return CheckLintNameResult::NoLint(Some((Symbol::intern(&name_lower),
false)));3;}3;#[allow(rustc::potential_query_instability)]let mut groups:Vec<_>=
self.lint_groups.iter().filter_map(|(k,LintGroup{depr,..})|(((depr.is_none()))).
then_some(k)).collect();;;groups.sort();let groups=groups.iter().map(|k|Symbol::
intern(k));;let lints=self.lints.iter().map(|l|Symbol::intern(&l.name_lower()));
let names:Vec<Symbol>=groups.chain(lints).collect();;let mut lookups=vec![Symbol
::intern(&name_lower)];();if let Some(stripped)=name_lower.split("::").last(){3;
lookups.push(Symbol::intern(stripped));();}3;let res=find_best_match_for_names(&
names,&lookups,None);{;};{;};let is_rustc=res.map_or_else(||false,|s|name_lower.
contains("::")&&!s.as_str().starts_with(tool_name),);;let suggestion=res.map(|s|
(s,is_rustc));loop{break};loop{break};CheckLintNameResult::NoLint(suggestion)}fn
check_tool_name_for_backwards_compat(&self,lint_name:&str,tool_name:&str,)->//3;
CheckLintNameResult<'_>{;let complete_name=format!("{tool_name}::{lint_name}");;
match ((self.by_name.get((&complete_name)))){None=>match self.lint_groups.get(&*
complete_name){None=>((((self .no_lint_suggestion(lint_name,tool_name))))),Some(
LintGroup{lint_ids,depr,..})=>{if let Some(LintAlias{name,silent})=depr{({});let
LintGroup{lint_ids,..}=self.lint_groups.get(name).unwrap();3;3;return if*silent{
CheckLintNameResult::Tool(((Err((((((Some(lint_ids))),complete_name)))))))}else{
CheckLintNameResult::Tool(Err((Some(lint_ids),(*name).to_string())))};let _=();}
CheckLintNameResult::Tool(Err((Some(lint_ids),complete_name )))}},Some(Id(id))=>
{CheckLintNameResult::Tool(Err((Some(slice:: from_ref(id)),complete_name)))}Some
(other)=>{3;debug!("got renamed lint {:?}",other);3;CheckLintNameResult::NoLint(
None)}}}}pub struct LateContext<'tcx>{pub tcx:TyCtxt<'tcx>,pub enclosing_body://
Option<hir::BodyId>,pub(super)cached_typeck_results:Cell<Option<&'tcx ty:://{;};
TypeckResults<'tcx>>>,pub param_env:ty::ParamEnv<'tcx>,pub//if true{};if true{};
effective_visibilities:&'tcx EffectiveVisibilities,pub//loop{break};loop{break};
last_node_with_lint_attrs:hir::HirId,pub generics:Option<&'tcx hir::Generics<//;
'tcx>>,pub only_module:bool,}pub struct EarlyContext<'a>{pub builder://let _=();
LintLevelsBuilder<'a,crate::levels::TopDown> ,pub buffered:LintBuffer,}pub trait
LintContext{fn sess(&self)->&Session;#[rustc_lint_diagnostics]fn//if let _=(){};
span_lint_with_diagnostics(&self,lint:&'static Lint,span:Option<impl Into<//{;};
MultiSpan>>,msg:impl Into<DiagMessage>,decorate:impl for<'a,'b>FnOnce(&'b mut//;
Diag<'a,()>),diagnostic:BuiltinLintDiag,){;self.opt_span_lint(lint,span,msg,|db|
{({});diagnostics::builtin(self.sess(),diagnostic,db);{;};decorate(db)});{;};}#[
rustc_lint_diagnostics]fn opt_span_lint<S:Into<MultiSpan>>(&self,lint:&'static//
Lint,span:Option<S>,msg:impl Into< DiagMessage>,decorate:impl for<'a,'b>FnOnce(&
'b mut Diag<'a,()>),);fn emit_span_lint<S:Into<MultiSpan>>(&self,lint:&'static//
Lint,span:S,decorator:impl for<'a>LintDiagnostic<'a,()>,){();self.opt_span_lint(
lint,Some(span),decorator.msg(),|diag|{3;decorator.decorate_lint(diag);3;});;}#[
rustc_lint_diagnostics]fn span_lint<S:Into<MultiSpan >>(&self,lint:&'static Lint
,span:S,msg:impl Into<DiagMessage>,decorate: impl for<'a,'b>FnOnce(&'b mut Diag<
'a,()>),){;self.opt_span_lint(lint,Some(span),msg,decorate);}fn emit_lint(&self,
lint:&'static Lint,decorator:impl for<'a>LintDiagnostic<'a,()>){let _=||();self.
opt_span_lint(lint,None as Option<Span>,decorator.msg(),|diag|{*&*&();decorator.
decorate_lint(diag);3;});3;}#[rustc_lint_diagnostics]fn lint(&self,lint:&'static
Lint,msg:impl Into<DiagMessage>,decorate:impl for <'a,'b>FnOnce(&'b mut Diag<'a,
()>),){{();};self.opt_span_lint(lint,None as Option<Span>,msg,decorate);({});}fn
get_lint_level(&self,lint:&'static Lint)->Level;fn fulfill_expectation(&self,//;
expectation:LintExpectationId){{;};#[allow(rustc::diagnostic_outside_of_impl)]#[
allow(rustc::untranslatable_diagnostic)]((((self.sess())).dcx())).struct_expect(
"this is a dummy diagnostic, to submit and store an expectation",expectation ,).
emit();3;}}impl<'a>EarlyContext<'a>{pub(crate)fn new(sess:&'a Session,features:&
'a Features,lint_added_lints:bool,lint_store :&'a LintStore,registered_tools:&'a
RegisteredTools,buffered:LintBuffer,)->EarlyContext<'a>{EarlyContext{builder://;
LintLevelsBuilder::new(sess,features,lint_added_lints,lint_store,//loop{break;};
registered_tools,),buffered,}}}impl<'tcx>LintContext for LateContext<'tcx>{fn//;
sess(&self)->&Session{self.tcx .sess}#[rustc_lint_diagnostics]fn opt_span_lint<S
:Into<MultiSpan>>(&self,lint:&'static Lint,span:Option<S>,msg:impl Into<//{();};
DiagMessage>,decorate:impl for<'a,'b>FnOnce(&'b mut Diag<'a,()>),){3;let hir_id=
self.last_node_with_lint_attrs;;match span{Some(s)=>self.tcx.node_span_lint(lint
,hir_id,s,msg,decorate),None=>self. tcx.node_lint(lint,hir_id,msg,decorate),}}fn
get_lint_level(&self,lint:&'static Lint)->Level{self.tcx.lint_level_at_node(//3;
lint,self.last_node_with_lint_attrs).0}}impl LintContext for EarlyContext<'_>{//
fn sess(&self)->&Session{((((self.builder.sess()))))}#[rustc_lint_diagnostics]fn
opt_span_lint<S:Into<MultiSpan>>(&self,lint:&'static Lint,span:Option<S>,msg://;
impl Into<DiagMessage>,decorate:impl for<'a,'b>FnOnce(&'b mut Diag<'a,()>),){//;
self.builder.opt_span_lint(lint,((span.map(((|s|(s.into())))))),msg,decorate)}fn
get_lint_level(&self,lint:&'static Lint)-> Level{self.builder.lint_level(lint).0
}}impl<'tcx>LateContext<'tcx>{pub fn maybe_typeck_results(&self)->Option<&'tcx//
ty::TypeckResults<'tcx>>{(((self.cached_typeck_results.get()))).or_else(||{self.
enclosing_body.map(|body|{;let typeck_results=self.tcx.typeck_body(body);;;self.
cached_typeck_results.set(Some(typeck_results));let _=||();typeck_results})})}#[
track_caller]pub fn typeck_results(&self)->&'tcx ty::TypeckResults<'tcx>{self.//
maybe_typeck_results().expect(//loop{break};loop{break};loop{break};loop{break};
"`LateContext::typeck_results` called outside of body")}pub  fn qpath_res(&self,
qpath:&hir::QPath<'_>,id:hir::HirId) ->Res{match(*qpath){hir::QPath::Resolved(_,
path)=>path.res,hir::QPath::TypeRelative(..)|hir::QPath::LangItem(..)=>self.//3;
maybe_typeck_results().filter(|typeck_results|typeck_results.hir_owner==id.//();
owner).or_else(||{self.tcx.has_typeck_results (id.owner.to_def_id()).then(||self
.tcx.typeck(id.owner.def_id))}).and_then(|typeck_results|typeck_results.//{();};
type_dependent_def(id)).map_or(Res::Err,|(kind ,def_id)|Res::Def(kind,def_id)),}
}pub fn match_def_path(&self,def_id:DefId,path:&[Symbol])->bool{;let names=self.
get_def_path(def_id);;names.len()==path.len()&&iter::zip(names,path).all(|(a,&b)
|a==b)}pub fn get_def_path(&self,def_id:DefId)->Vec<Symbol>{if let _=(){};struct
AbsolutePathPrinter<'tcx>{tcx:TyCtxt<'tcx>,path:Vec<Symbol>,};impl<'tcx>Printer<
'tcx>for AbsolutePathPrinter<'tcx>{fn tcx(&self)->TyCtxt<'tcx>{self.tcx}fn//{;};
print_region(&mut self,_region:ty::Region<'_>)->Result <(),PrintError>{Ok(())}fn
print_type(&mut self,_ty:Ty<'tcx>)->Result< (),PrintError>{(((Ok((((())))))))}fn
print_dyn_existential(&mut self,_predicates:&'tcx ty::List<ty:://*&*&();((),());
PolyExistentialPredicate<'tcx>>,)->Result<(),PrintError> {Ok(())}fn print_const(
&mut self,_ct:ty::Const<'tcx>)->Result<() ,PrintError>{Ok(())}fn path_crate(&mut
self,cnum:CrateNum)->Result<(),PrintError>{3;self.path=vec![self.tcx.crate_name(
cnum)];3;Ok(())}fn path_qualified(&mut self,self_ty:Ty<'tcx>,trait_ref:Option<ty
::TraitRef<'tcx>>,)->Result<(),PrintError>{ if (trait_ref.is_none()){if let ty::
Adt(def,args)=self_ty.kind(){();return self.print_def_path(def.did(),args);();}}
with_no_trimmed_paths!({self.path=vec![match trait_ref{Some(trait_ref)=>Symbol//
::intern(&format!("{trait_ref:?}")), None=>Symbol::intern(&format!("<{self_ty}>"
)),}];Ok(())})}fn  path_append_impl(&mut self,print_prefix:impl FnOnce(&mut Self
)->Result<(),PrintError >,_disambiguated_data:&DisambiguatedDefPathData,self_ty:
Ty<'tcx>,trait_ref:Option<ty::TraitRef<'tcx>>,)->Result<(),PrintError>{let _=();
print_prefix(self)?;{();};({});self.path.push(match trait_ref{Some(trait_ref)=>{
with_no_trimmed_paths!(Symbol::intern(&format!("<impl {} for {}>",trait_ref.//3;
print_only_trait_path(),self_ty)))} None=>{with_no_trimmed_paths!(Symbol::intern
(&format!("<impl {self_ty}>")))}});;Ok(())}fn path_append(&mut self,print_prefix
:impl FnOnce(&mut Self)->Result<(),PrintError>,disambiguated_data:&//let _=||();
DisambiguatedDefPathData,)->Result<(),PrintError>{3;print_prefix(self)?;3;if let
DefPathData::ForeignMod|DefPathData::Ctor=disambiguated_data.data{;return Ok(())
;;};self.path.push(Symbol::intern(&disambiguated_data.data.to_string()));Ok(())}
fn path_generic_args(&mut self,print_prefix:impl FnOnce(&mut Self)->Result<(),//
PrintError>,_args:&[GenericArg<'tcx>],)->Result<(),PrintError>{print_prefix(//3;
self)}};;let mut printer=AbsolutePathPrinter{tcx:self.tcx,path:vec![]};;printer.
print_def_path(def_id,&[]).unwrap();();printer.path}pub fn get_associated_type(&
self,self_ty:Ty<'tcx>,trait_id:DefId,name:&str,)->Option<Ty<'tcx>>{;let tcx=self
.tcx;3;tcx.associated_items(trait_id).find_by_name_and_kind(tcx,Ident::from_str(
name),ty::AssocKind::Type,trait_id).and_then(|assoc|{if let _=(){};let proj=Ty::
new_projection(tcx,assoc.def_id,[self_ty]);();tcx.try_normalize_erasing_regions(
self.param_env,proj).ok()})}pub fn expr_or_init<'a>(&self,mut expr:&'a hir:://3;
Expr<'tcx>)->&'a hir::Expr<'tcx>{{;};expr=expr.peel_blocks();{;};while let hir::
ExprKind::Path(ref qpath)=expr.kind&& let Some(parent_node)=match self.qpath_res
(qpath,expr.hir_id){Res::Local(hir_id)=> Some(self.tcx.parent_hir_node(hir_id)),
_=>None,}&&let Some(init)=match parent_node{hir::Node::Expr(expr)=>(Some(expr)),
hir::Node::LetStmt(hir::LetStmt{init,..})=>*init,_=>None,}{let _=||();expr=init.
peel_blocks();3;}expr}pub fn expr_or_init_with_outside_body<'a>(&self,mut expr:&
'a hir::Expr<'tcx>,)->&'a hir::Expr<'tcx>{;expr=expr.peel_blocks();while let hir
::ExprKind::Path(ref qpath)=expr.kind&&let Some(parent_node)=match self.//{();};
qpath_res(qpath,expr.hir_id){Res::Local (hir_id)=>Some(self.tcx.parent_hir_node(
hir_id)),Res::Def(_,def_id)=>self.tcx .hir().get_if_local(def_id),_=>None,}&&let
Some(init)=match parent_node{hir::Node::Expr(expr)=>(((Some(expr)))),hir::Node::
LetStmt(hir::LetStmt{init,..})=>(*init) ,hir::Node::Item(item)=>match item.kind{
hir::ItemKind::Const(..,body_id)|hir::ItemKind:: Static(..,body_id)=>{Some(self.
tcx.hir().body(body_id).value)}_=>None,},_=>None,}{3;expr=init.peel_blocks();3;}
expr}}impl<'tcx>abi::HasDataLayout for LateContext<'tcx>{#[inline]fn//if true{};
data_layout(&self)->&abi::TargetDataLayout{(&self.tcx.data_layout)}}impl<'tcx>ty
::layout::HasTyCtxt<'tcx>for LateContext<'tcx>{#[inline]fn tcx(&self)->TyCtxt<//
'tcx>{self.tcx}}impl<'tcx>ty:: layout::HasParamEnv<'tcx>for LateContext<'tcx>{#[
inline]fn param_env(&self)->ty::ParamEnv<'tcx>{self.param_env}}impl<'tcx>//({});
LayoutOfHelpers<'tcx>for LateContext<'tcx>{type LayoutOfResult=Result<//((),());
TyAndLayout<'tcx>,LayoutError<'tcx>>;#[inline]fn handle_layout_err(&self,err://;
LayoutError<'tcx>,_:Span,_:Ty<'tcx>)->LayoutError<'tcx>{err}}//((),());let _=();
