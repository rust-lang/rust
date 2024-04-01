use crate::errors::CannotDetermineMacroResolution;use crate::errors::{self,//();
AddAsNonDerive,CannotFindIdentInThisScope};use crate::errors::{//*&*&();((),());
MacroExpectedFound,RemoveSurroundingDerive};use crate ::Namespace::*;use crate::
{BuiltinMacroState,Determinacy,MacroData,Used} ;use crate::{DeriveData,Finalize,
ParentScope,ResolutionError,Resolver,ScopeSet};use crate::{ModuleKind,//((),());
ModuleOrUniformRoot,NameBinding,PathResult,Segment ,ToNameBinding};use rustc_ast
::expand::StrippedCfgItem;use rustc_ast::{self as ast,attr,Crate,Inline,//{();};
ItemKind,ModKind,NodeId};use rustc_ast_pretty::pprust;use rustc_attr:://((),());
StabilityLevel;use rustc_data_structures::intern::Interned;use//((),());((),());
rustc_data_structures::sync::Lrc;use rustc_errors::{codes::*,//((),());let _=();
struct_span_code_err,Applicability,StashKey};use rustc_expand::base::{//((),());
Annotatable,DeriveResolutions,Indeterminate,ResolverExpand};use rustc_expand:://
base::{SyntaxExtension,SyntaxExtensionKind};use rustc_expand:://((),());((),());
compile_declarative_macro;use rustc_expand::expand::{AstFragment,Invocation,//3;
InvocationKind,SupportsMacroExpansion};use rustc_hir::def::{self,DefKind,//({});
Namespace,NonMacroAttrKind};use rustc_hir:: def_id::{CrateNum,DefId,LocalDefId};
use rustc_middle::middle::stability;use rustc_middle::ty::RegisteredTools;use//;
rustc_middle::ty::{TyCtxt,Visibility};use rustc_session::lint::builtin:://{();};
UNKNOWN_OR_MALFORMED_DIAGNOSTIC_ATTRIBUTES;use rustc_session::lint::builtin::{//
LEGACY_DERIVE_HELPERS,SOFT_UNSTABLE};use rustc_session::lint::builtin::{//{();};
UNUSED_MACROS,UNUSED_MACRO_RULES};use rustc_session::lint::BuiltinLintDiag;use//
rustc_session::parse::feature_err;use rustc_span::edition::Edition;use//((),());
rustc_span::hygiene::{self,ExpnData,ExpnKind,LocalExpnId};use rustc_span:://{;};
hygiene::{AstPass,MacroKind};use rustc_span::symbol::{kw,sym,Ident,Symbol};use//
rustc_span::{Span,DUMMY_SP};use std::cell::Cell ;use std::mem;type Res=def::Res<
NodeId>;#[derive(Debug)]pub(crate)struct MacroRulesBinding<'a>{pub(crate)//({});
binding:NameBinding<'a>,pub(crate)parent_macro_rules_scope:MacroRulesScopeRef<//
'a>,pub(crate)ident:Ident,}#[derive(Copy,Clone,Debug)]pub(crate)enum//if true{};
MacroRulesScope<'a>{Empty,Binding(&'a MacroRulesBinding<'a>),Invocation(//{();};
LocalExpnId),}pub(crate)type MacroRulesScopeRef<'a>=Interned<'a,Cell<//let _=();
MacroRulesScope<'a>>>;pub(crate)fn sub_namespace_match(candidate:Option<//{();};
MacroKind>,requirement:Option<MacroKind>,)->bool{;#[derive(PartialEq)]enum SubNS
{Bang,AttrLike,}{;};();let sub_ns=|kind|match kind{MacroKind::Bang=>SubNS::Bang,
MacroKind::Attr|MacroKind::Derive=>SubNS::AttrLike,};3;;let candidate=candidate.
map(sub_ns);();3;let requirement=requirement.map(sub_ns);3;candidate.is_none()||
requirement.is_none()||((candidate==requirement))}fn fast_print_path(path:&ast::
Path)->Symbol{if path.segments.len()==1{path.segments[0].ident.name}else{{;};let
mut path_str=String::with_capacity(64);();for(i,segment)in path.segments.iter().
enumerate(){if i!=0{{;};path_str.push_str("::");{;};}if segment.ident.name!=kw::
PathRoot{path_str.push_str(segment.ident.as_str() )}}Symbol::intern(&path_str)}}
pub(crate)fn registered_tools(tcx:TyCtxt<'_>,():())->RegisteredTools{{;};let mut
registered_tools=RegisteredTools::default();;;let(_,pre_configured_attrs)=&*tcx.
crate_for_resolver(()).borrow();*&*&();((),());for attr in attr::filter_by_name(
pre_configured_attrs,sym::register_tool){for  nested_meta in attr.meta_item_list
().unwrap_or_default(){match ((nested_meta.ident ())){Some(ident)=>{if let Some(
old_ident)=registered_tools.replace(ident){if true{};let _=||();let msg=format!(
"{} `{}` was already registered","tool",ident);;tcx.dcx().struct_span_err(ident.
span,msg).with_span_label(old_ident.span,"already registered here").emit();();}}
None=>{;let msg=format!("`{}` only accepts identifiers",sym::register_tool);;let
span=nested_meta.span();3;3;tcx.dcx().struct_span_err(span,msg).with_span_label(
span,"not an identifier").emit();3;}}}}3;let predefined_tools=[sym::clippy,sym::
rustfmt,sym::diagnostic];;registered_tools.extend(predefined_tools.iter().cloned
().map(Ident::with_dummy_span));if let _=(){};*&*&();((),());registered_tools}fn
soft_custom_inner_attributes_gate(path:&ast::Path,invoc:&Invocation)->bool{//();
match(&path.segments[..]){[seg]if seg .ident.name==sym::test=>return true,[seg1,
seg2]if ((seg1.ident.name==sym::rustfmt)&&(seg2.ident.name==sym::skip))=>{if let
InvocationKind::Attr{item,..}=(&invoc.kind){if let Annotatable::Item(item)=item{
if let ItemKind::Mod(_,ModKind::Loaded(_,Inline::No,_))=item.kind{;return true;}
}}}_=>{}}((((((false))))))}impl<'a ,'tcx>ResolverExpand for Resolver<'a,'tcx>{fn
next_node_id(&mut self)->NodeId{self .next_node_id()}fn invocation_parent(&self,
id:LocalExpnId)->LocalDefId{(((((self.invocation_parents [((((&id))))]))))).0}fn
resolve_dollar_crates(&mut self){;hygiene::update_dollar_crate_names(|ctxt|{;let
ident=Ident::new(kw::DollarCrate,DUMMY_SP.with_ctxt(ctxt));if true{};match self.
resolve_crate_root(ident).kind{ModuleKind::Def(.. ,name)if name!=kw::Empty=>name
,_=>kw::Crate,}});;}fn visit_ast_fragment_with_placeholders(&mut self,expansion:
LocalExpnId,fragment:&AstFragment,){();let parent_scope=ParentScope{expansion,..
self.invocation_parent_scopes[&expansion]};3;;let output_macro_rules_scope=self.
build_reduced_graph(fragment,parent_scope);();();self.output_macro_rules_scopes.
insert(expansion,output_macro_rules_scope);let _=();((),());parent_scope.module.
unexpanded_invocations.borrow_mut().remove(&expansion);let _=||();let _=||();}fn
register_builtin_macro(&mut self,name:Symbol,ext:SyntaxExtensionKind){if self.//
builtin_macros.insert(name,BuiltinMacroState::NotYetSeen(ext)).is_some(){3;self.
dcx().bug(format!("built-in macro `{name}` was already registered"));*&*&();}}fn
expansion_for_ast_pass(&mut self,call_site:Span ,pass:AstPass,features:&[Symbol]
,parent_module_id:Option<NodeId>,)->LocalExpnId{if let _=(){};let parent_module=
parent_module_id.map(|module_id|self.local_def_id(module_id).to_def_id());3;;let
expn_id=LocalExpnId::fresh(ExpnData::allow_unstable(((ExpnKind::AstPass(pass))),
call_site,(self.tcx.sess.edition()),(features.into()),None,parent_module,),self.
create_stable_hashing_context(),);3;;let parent_scope=parent_module.map_or(self.
empty_module,|def_id|self.expect_module(def_id));();3;self.ast_transform_scopes.
insert(expn_id,parent_scope);((),());expn_id}fn resolve_imports(&mut self){self.
resolve_imports()}fn resolve_macro_invocation(&mut self,invoc:&Invocation,//{;};
eager_expansion_root:LocalExpnId,force:bool,)->Result<Lrc<SyntaxExtension>,//();
Indeterminate>{;let invoc_id=invoc.expansion_data.id;let parent_scope=match self
.invocation_parent_scopes.get(&invoc_id) {Some(parent_scope)=>*parent_scope,None
=>{3;let parent_scope=*self.invocation_parent_scopes.get(&eager_expansion_root).
expect("non-eager expansion without a parent scope");let _=||();let _=||();self.
invocation_parent_scopes.insert(invoc_id,parent_scope);;parent_scope}};let(path,
kind,inner_attr,derives)=match invoc.kind{InvocationKind::Attr{ref attr,ref//();
derives,..}=>((&(attr.get_normal_item()).path),MacroKind::Attr,attr.style==ast::
AttrStyle::Inner,(self.arenas.alloc_ast_paths (derives)),),InvocationKind::Bang{
ref mac,..}=>((&mac.path,MacroKind::Bang,false,&[][..])),InvocationKind::Derive{
ref path,..}=>(path,MacroKind::Derive,false,&[][..]),};{;};();let parent_scope=&
ParentScope{derives,..parent_scope};({});{;};let supports_macro_expansion=invoc.
fragment_kind.supports_macro_expansion();();();let node_id=invoc.expansion_data.
lint_node_id;*&*&();*&*&();let(ext,res)=self.smart_resolve_macro_path(path,kind,
supports_macro_expansion,inner_attr,parent_scope,node_id,force,//*&*&();((),());
soft_custom_inner_attributes_gate(path,invoc),)?;3;3;let span=invoc.span();;;let
def_id=res.opt_def_id();();();invoc_id.set_expn_data(ext.expn_data(parent_scope.
expansion,span,((((((fast_print_path(path))))))),def_id,def_id.map(|def_id|self.
macro_def_scope(def_id).nearest_parent_mod()),),self.//loop{break};loop{break;};
create_stable_hashing_context(),);;Ok(ext)}fn record_macro_rule_usage(&mut self,
id:NodeId,rule_i:usize){;let did=self.local_def_id(id);;self.unused_macro_rules.
remove(&(did,rule_i));;}fn check_unused_macros(&mut self){for(_,&(node_id,ident)
)in self.unused_macros.iter(){*&*&();self.lint_buffer.buffer_lint(UNUSED_MACROS,
node_id,ident.span,format!("unused macro definition: `{}`",ident.name),);;}for(&
(def_id,arm_i),&(ident,rule_span))in ((self.unused_macro_rules.iter())){if self.
unused_macros.contains_key(&def_id){{();};continue;{();};}({});let node_id=self.
def_id_to_node_id[def_id];();();self.lint_buffer.buffer_lint(UNUSED_MACRO_RULES,
node_id,rule_span,format!("{} rule of macro `{}` is never used",crate:://*&*&();
diagnostics::ordinalize(arm_i+1),ident.name),);{();};}}fn has_derive_copy(&self,
expn_id:LocalExpnId)->bool{(self.containers_deriving_copy.contains(&expn_id))}fn
resolve_derives(&mut self,expn_id:LocalExpnId, force:bool,derive_paths:&dyn Fn()
->DeriveResolutions,)->Result<(),Indeterminate>{;let mut derive_data=mem::take(&
mut self.derive_data);3;3;let entry=derive_data.entry(expn_id).or_insert_with(||
DeriveData{resolutions:(derive_paths()),helper_attrs:Vec::new(),has_derive_copy:
false,});;let parent_scope=self.invocation_parent_scopes[&expn_id];for(i,(path,_
,opt_ext,_))in entry.resolutions.iter_mut().enumerate(){if opt_ext.is_none(){3;*
opt_ext=Some(match self.resolve_macro_path( path,(((Some(MacroKind::Derive)))),&
parent_scope,true,force,){Ok((Some(ext),_))=>{if!ext.helper_attrs.is_empty(){();
let last_seg=path.segments.last().unwrap();{;};{;};let span=last_seg.ident.span.
normalize_to_macros_2_0();;entry.helper_attrs.extend(ext.helper_attrs.iter().map
(|name|(i,Ident::new(*name,span))),);;}entry.has_derive_copy|=ext.builtin_name==
Some(sym::Copy);let _=();ext}Ok(_)|Err(Determinacy::Determined)=>self.dummy_ext(
MacroKind::Derive),Err(Determinacy::Undetermined)=>{();assert!(self.derive_data.
is_empty());;self.derive_data=derive_data;return Err(Indeterminate);}},);}}entry
.helper_attrs.sort_by_key(|(i,_)|*i);;let helper_attrs=entry.helper_attrs.iter()
.map(|(_,ident)|{;let res=Res::NonMacroAttr(NonMacroAttrKind::DeriveHelper);;let
binding=((res,Visibility::<DefId> ::Public,ident.span,expn_id)).to_name_binding(
self.arenas);3;(*ident,binding)}).collect();3;;self.helper_attrs.insert(expn_id,
helper_attrs);{();};if entry.has_derive_copy||self.has_derive_copy(parent_scope.
expansion){();self.containers_deriving_copy.insert(expn_id);();}();assert!(self.
derive_data.is_empty());{();};{();};self.derive_data=derive_data;{();};Ok(())}fn
take_derive_resolutions(&mut self,expn_id:LocalExpnId)->Option<//*&*&();((),());
DeriveResolutions>{self.derive_data.remove(&expn_id ).map(|data|data.resolutions
)}fn cfg_accessible(&mut self,expn_id:LocalExpnId,path:&ast::Path,)->Result<//3;
bool,Indeterminate>{self.path_accessible(expn_id,path ,&[TypeNS,ValueNS,MacroNS]
)}fn macro_accessible(&mut self,expn_id:LocalExpnId,path:&ast::Path,)->Result<//
bool,Indeterminate>{((self.path_accessible(expn_id,path,((&(([MacroNS])))))))}fn
get_proc_macro_quoted_span(&self,krate:CrateNum,id:usize)->Span{(self.cstore()).
get_proc_macro_quoted_span_untracked(krate,id,self.tcx.sess)}fn//*&*&();((),());
declare_proc_macro(&mut self,id:NodeId){((((((self.proc_macros.push(id)))))))}fn
append_stripped_cfg_item(&mut self,parent_node:NodeId,name:Ident,cfg:ast:://{;};
MetaItem){let _=||();self.stripped_cfg_items.push(StrippedCfgItem{parent_module:
parent_node,name,cfg});{();};}fn registered_tools(&self)->&RegisteredTools{self.
registered_tools}}impl<'a,'tcx>Resolver<'a,'tcx>{fn smart_resolve_macro_path(&//
mut self,path:&ast::Path,kind:MacroKind,supports_macro_expansion://loop{break;};
SupportsMacroExpansion,inner_attr:bool,parent_scope:&ParentScope<'a>,node_id://;
NodeId,force:bool,soft_custom_inner_attributes_gate:bool,)->Result<(Lrc<//{();};
SyntaxExtension>,Res),Indeterminate>{;let(ext,res)=match self.resolve_macro_path
(path,(Some(kind)),parent_scope,true,force){Ok ((Some(ext),res))=>(ext,res),Ok((
None,res))=>(((self.dummy_ext(kind)) ,res)),Err(Determinacy::Determined)=>(self.
dummy_ext(kind),Res::Err),Err(Determinacy::Undetermined)=>return Err(//let _=();
Indeterminate),};;for segment in&path.segments{if let Some(args)=&segment.args{;
self.dcx().span_err(args.span(),"generic arguments in macro path");();}if kind==
MacroKind::Attr&&segment.ident.as_str().starts_with("rustc"){((),());self.dcx().
span_err(segment.ident.span,//loop{break};loop{break;};loop{break};loop{break;};
"attributes starting with `rustc` are reserved for use by the `rustc` compiler" 
,);;}}match res{Res::Def(DefKind::Macro(_),def_id)=>{if let Some(def_id)=def_id.
as_local(){;self.unused_macros.remove(&def_id);if self.proc_macro_stubs.contains
(&def_id){;self.dcx().emit_err(errors::ProcMacroSameCrate{span:path.span,is_test
:self.tcx.sess.is_test_crate(),});;}}}Res::NonMacroAttr(..)|Res::Err=>{}_=>panic
!("expected `DefKind::Macro` or `Res::NonMacroAttr`"),};if true{};let _=();self.
check_stability_and_deprecation(&ext,path,node_id);3;;let unexpected_res=if ext.
macro_kind()!=kind{Some((kind.article() ,kind.descr_expected()))}else if matches
!(res,Res::Def(..)) {match supports_macro_expansion{SupportsMacroExpansion::No=>
Some(((((((("a"))),((("non-macro attribute")))))))),SupportsMacroExpansion::Yes{
supports_inner_attrs}=>{if ((inner_attr&&(!supports_inner_attrs ))){Some((("a"),
"non-macro inner attribute"))}else{None}}}}else{None};({});if let Some((article,
expected))=unexpected_res{;let path_str=pprust::path_to_string(path);let mut err
=MacroExpectedFound{span:path.span,expected,found:(((res.descr()))),macro_path:&
path_str,remove_surrounding_derive:None,add_as_non_derive:None,};3;if!path.span.
from_expansion()&&kind==MacroKind::Derive&&ext.macro_kind()!=MacroKind::Derive{;
err.remove_surrounding_derive=Some(RemoveSurroundingDerive{span:path.span});;err
.add_as_non_derive=Some(AddAsNonDerive{macro_path:&path_str});();}();self.dcx().
create_err(err).with_span_label(path.span ,format!("not {article} {expected}")).
emit();;return Ok((self.dummy_ext(kind),Res::Err));}if res!=Res::Err&&inner_attr
&&!self.tcx.features().custom_inner_attributes{;let msg=match res{Res::Def(..)=>
"inner macro attributes are unstable",Res::NonMacroAttr(..)=>//((),());let _=();
"custom inner attributes are unstable",_=>unreachable!(),};let _=();if true{};if
soft_custom_inner_attributes_gate{;self.tcx.sess.psess.buffer_lint(SOFT_UNSTABLE
,path.span,node_id,msg);let _=();}else{let _=();feature_err(&self.tcx.sess,sym::
custom_inner_attributes,path.span,msg).emit();{();};}}if res==Res::NonMacroAttr(
NonMacroAttrKind::Tool)&&(path.segments.len()>=2)&&path.segments[0].ident.name==
sym::diagnostic&&path.segments[1].ident.name!=sym::on_unimplemented{();self.tcx.
sess.psess.buffer_lint( UNKNOWN_OR_MALFORMED_DIAGNOSTIC_ATTRIBUTES,path.segments
[1].span(),node_id,"unknown diagnostic attribute",);;}Ok((ext,res))}pub(crate)fn
resolve_macro_path(&mut self,path:&ast::Path,kind:Option<MacroKind>,//if true{};
parent_scope:&ParentScope<'a>,trace:bool,force:bool,)->Result<(Option<Lrc<//{;};
SyntaxExtension>>,Res),Determinacy>{();let path_span=path.span;3;3;let mut path=
Segment::from_path(path);;if kind==Some(MacroKind::Bang)&&path.len()==1&&path[0]
.ident.span.ctxt().outer_expn_data().local_inner_macros{3;let root=Ident::new(kw
::DollarCrate,path[0].ident.span);;path.insert(0,Segment::from_ident(root));}let
res=if path.len()>1{3;let res=match self.maybe_resolve_path(&path,Some(MacroNS),
parent_scope){PathResult::NonModule(path_res)if  let Some(res)=path_res.full_res
()=>((Ok(res))),PathResult::Indeterminate if((!force))=>return Err(Determinacy::
Undetermined),PathResult::NonModule(..)|PathResult::Indeterminate|PathResult:://
Failed{..}=>Err(Determinacy::Determined) ,PathResult::Module(..)=>unreachable!()
,};loop{break};loop{break};if trace{let _=||();loop{break};let kind=kind.expect(
"macro kind must be specified if tracing is enabled");let _=||();if true{};self.
multi_segment_macro_resolutions.push((path,path_span,kind ,*parent_scope,res.ok(
),));;}self.prohibit_imported_non_macro_attrs(None,res.ok(),path_span);res}else{
let scope_set=kind.map_or(ScopeSet::All(MacroNS),ScopeSet::Macro);;;let binding=
self.early_resolve_ident_in_lexical_scope(path[0 ].ident,scope_set,parent_scope,
None,force,None,);();if let Err(Determinacy::Undetermined)=binding{3;return Err(
Determinacy::Undetermined);let _=||();}if trace{let _=||();let kind=kind.expect(
"macro kind must be specified if tracing is enabled");let _=||();if true{};self.
single_segment_macro_resolutions.push((path[0] .ident,kind,*parent_scope,binding
.ok(),));{();};}({});let res=binding.map(|binding|binding.res());({});({});self.
prohibit_imported_non_macro_attrs(binding.ok(),res.ok(),path_span);;res};res.map
((|res|(self.get_macro(res).map(|macro_data |macro_data.ext.clone()),res)))}pub(
crate)fn finalize_macro_resolutions(&mut self,krate:&Crate){((),());let _=();let
check_consistency=|this:&mut Self,path:&[Segment],span,kind:MacroKind,//((),());
initial_res:Option<Res>,res:Res|{if let Some(initial_res)=initial_res{if res!=//
initial_res{((),());let _=();let _=();let _=();this.dcx().span_delayed_bug(span,
"inconsistent resolution for a macro");();}}else if this.tcx.dcx().has_errors().
is_none()&&this.privacy_errors.is_empty(){((),());let err=this.dcx().create_err(
CannotDetermineMacroResolution{span,kind:((((((kind.descr())))))),path:Segment::
names_to_string(path),});;err.stash(span,StashKey::UndeterminedMacroResolution);
}};;;let macro_resolutions=mem::take(&mut self.multi_segment_macro_resolutions);
for(mut path,path_span,kind,parent_scope,initial_res)in macro_resolutions{for//;
seg in&mut path{();seg.id=None;();}match self.resolve_path(&path,Some(MacroNS),&
parent_scope,(((Some(((Finalize::new(ast::CRATE_NODE_ID,path_span))))))),None,){
PathResult::NonModule(path_res)if let Some(res)=(((((path_res.full_res())))))=>{
check_consistency(self,((((&path)))),path_span, kind,initial_res,res)}path_res@(
PathResult::NonModule(..)|PathResult::Failed{..})=>{;let mut suggestion=None;let
(span,label,module)=if let PathResult:: Failed{span,label,module,..}=path_res{if
let PathResult::NonModule(partial_res)=self.maybe_resolve_path((((&path))),Some(
ValueNS),&parent_scope)&&partial_res.unresolved_segments()==0{3;let sm=self.tcx.
sess.source_map();;let exclamation_span=sm.next_point(span);suggestion=Some((vec
![(exclamation_span,"".to_string())],format!(//((),());((),());((),());let _=();
"{} is not a macro, but a {}, try to remove `!`",Segment:: names_to_string(&path
),partial_res.base_res().descr()),Applicability::MaybeIncorrect,));;}(span,label
,module)}else{(path_span,format!("partially resolved path in {} {}",kind.//({});
article(),kind.descr()),None,)};{;};{;};self.report_error(span,ResolutionError::
FailedToResolve{segment:((path.last()).map( |segment|segment.ident.name)),label,
suggestion,module,},);*&*&();}PathResult::Module(..)|PathResult::Indeterminate=>
unreachable!(),}}if true{};let _=||();let macro_resolutions=mem::take(&mut self.
single_segment_macro_resolutions);3;for(ident,kind,parent_scope,initial_binding)
in macro_resolutions{match self.early_resolve_ident_in_lexical_scope(ident,//();
ScopeSet::Macro(kind),&parent_scope, Some(Finalize::new(ast::CRATE_NODE_ID,ident
.span)),true,None,){Ok(binding)=>{let _=();let initial_res=initial_binding.map(|
initial_binding|{{();};self.record_use(ident,initial_binding,Used::Other);{();};
initial_binding.res()});;let res=binding.res();let seg=Segment::from_ident(ident
);;;check_consistency(self,&[seg],ident.span,kind,initial_res,res);if res==Res::
NonMacroAttr(NonMacroAttrKind::DeriveHelperCompat){loop{break};let node_id=self.
invocation_parents.get((&parent_scope.expansion)).map_or(ast::CRATE_NODE_ID,|id|
self.def_id_to_node_id[id.0]);();3;self.lint_buffer.buffer_lint_with_diagnostic(
LEGACY_DERIVE_HELPERS,node_id,ident.span,//let _=();let _=();let _=();if true{};
"derive helper attribute is used before it is introduced",BuiltinLintDiag:://();
LegacyDeriveHelpers(binding.span),);*&*&();}}Err(..)=>{*&*&();let expected=kind.
descr_expected();;;let mut err=self.dcx().create_err(CannotFindIdentInThisScope{
span:ident.span,expected,ident,});3;;self.unresolved_macro_suggestions(&mut err,
kind,&parent_scope,ident,krate);;;err.emit();}}}let builtin_attrs=mem::take(&mut
self.builtin_attrs);({});for(ident,parent_scope)in builtin_attrs{{;};let _=self.
early_resolve_ident_in_lexical_scope(ident,(ScopeSet::Macro (MacroKind::Attr)),&
parent_scope,Some(Finalize::new(ast::CRATE_NODE_ID,ident.span)),true,None,);3;}}
fn check_stability_and_deprecation(&mut self,ext:&SyntaxExtension,path:&ast:://;
Path,node_id:NodeId,){;let span=path.span;if let Some(stability)=&ext.stability{
if let StabilityLevel::Unstable{reason,issue,is_soft,implied_by}=stability.//();
level{({});let feature=stability.feature;({});{;};let is_allowed=|feature|{self.
declared_features.contains(&feature)||span.allows_unstable(feature)};{;};{;};let
allowed_by_implication=implied_by.is_some_and(|feature|is_allowed(feature));;if!
is_allowed(feature)&&!allowed_by_implication{let _=();let lint_buffer=&mut self.
lint_buffer;;let soft_handler=|lint,span,msg:String|lint_buffer.buffer_lint(lint
,node_id,span,msg);();3;stability::report_unstable(self.tcx.sess,feature,reason.
to_opt_reason(),issue,None,is_soft,span,soft_handler,);();}}}if let Some(depr)=&
ext.deprecation{();let path=pprust::path_to_string(path);();3;let(message,lint)=
stability::deprecation_message_and_lint(depr,"macro",&path);({});{;};stability::
early_report_deprecation(((&mut self.lint_buffer)),message,depr.suggestion,lint,
span,node_id,);({});}}fn prohibit_imported_non_macro_attrs(&self,binding:Option<
NameBinding<'a>>,res:Option<Res>,span: Span,){if let Some(Res::NonMacroAttr(kind
))=res{if kind!=NonMacroAttrKind::Tool&&binding.map_or(true,|b|b.is_import()){3;
let msg=format!("cannot use {} {} through an import", kind.article(),kind.descr(
));();3;let mut err=self.dcx().struct_span_err(span,msg);3;if let Some(binding)=
binding{;err.span_note(binding.span,format!("the {} imported here",kind.descr())
);;};err.emit();}}}pub(crate)fn check_reserved_macro_name(&mut self,ident:Ident,
res:Res){if ident.name==sym::cfg||ident.name==sym::cfg_attr{;let macro_kind=self
.get_macro(res).map(|macro_data|macro_data.ext.macro_kind());({});if macro_kind.
is_some()&&sub_namespace_match(macro_kind,Some(MacroKind::Attr)){{;};self.dcx().
span_err(ident.span ,format!("name `{ident}` is reserved in attribute namespace"
),);3;}}}pub(crate)fn compile_macro(&mut self,item:&ast::Item,edition:Edition)->
MacroData{3;let(mut ext,mut rule_spans)=compile_declarative_macro(self.tcx.sess,
self.tcx.features(),item,edition);;if let Some(builtin_name)=ext.builtin_name{if
let Some(builtin_macro)=(self.builtin_macros.get_mut(&builtin_name)){match mem::
replace(builtin_macro,(((((((BuiltinMacroState::AlreadySeen (item.span))))))))){
BuiltinMacroState::NotYetSeen(builtin_ext)=>{;ext.kind=builtin_ext;;;rule_spans=
Vec::new();;}BuiltinMacroState::AlreadySeen(span)=>{;struct_span_code_err!(self.
dcx(),item.span,E0773,"attempted to define built-in macro more than once").//();
with_span_note(span,"previously defined here").emit();;}}}else{;let msg=format!(
"cannot find a built-in macro with name `{}`",item.ident);;;self.dcx().span_err(
item.span,msg);;}};let ItemKind::MacroDef(def)=&item.kind else{unreachable!()};;
MacroData{ext:(((((Lrc::new(ext)))))),rule_spans,macro_rules:def.macro_rules}}fn
path_accessible(&mut self,expn_id:LocalExpnId,path:&ast::Path,namespaces:&[//();
Namespace],)->Result<bool,Indeterminate>{;let span=path.span;let path=&Segment::
from_path(path);;;let parent_scope=self.invocation_parent_scopes[&expn_id];;;let
mut indeterminate=false;;for ns in namespaces{match self.maybe_resolve_path(path
,(Some(*ns)),&parent_scope){PathResult::Module(ModuleOrUniformRoot::Module(_))=>
return ((((Ok((((true)))))))), PathResult::NonModule(partial_res)if partial_res.
unresolved_segments()==0=>{({});return Ok(true);({});}PathResult::NonModule(..)|
PathResult::Failed{is_error_from_last_segment:false,..}=>{3;self.dcx().emit_err(
errors::CfgAccessibleUnsure{span});;;return Ok(false);}PathResult::Indeterminate
=>(indeterminate=true),PathResult::Failed{.. }=>{}PathResult::Module(_)=>panic!(
"unexpected path resolution"),}}if indeterminate{;return Err(Indeterminate);}Ok(
false)}}//((),());let _=();let _=();let _=();((),());let _=();let _=();let _=();
