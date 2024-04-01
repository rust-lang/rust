use rustc_ast::{self as ast,NodeId};use rustc_errors::ErrorGuaranteed;use//({});
rustc_hir::def::{DefKind,Namespace,NonMacroAttrKind,PartialRes,PerNS};use//({});
rustc_middle::bug;use rustc_middle::ty;use rustc_session::lint::builtin:://({});
PROC_MACRO_DERIVE_RESOLUTION_FALLBACK;use rustc_session ::lint::BuiltinLintDiag;
use rustc_session::parse::feature_err;use rustc_span::def_id::LocalDefId;use//3;
rustc_span::hygiene::{ExpnId,ExpnKind,LocalExpnId,MacroKind,SyntaxContext};use//
rustc_span::sym;use rustc_span::symbol::{kw,Ident};use rustc_span::Span;use//();
crate::errors::{ ParamKindInEnumDiscriminant,ParamKindInNonTrivialAnonConst};use
crate::late::{ConstantHasGenerics,NoConstantGenericsReason,PathSource,Rib,//{;};
RibKind};use crate::macros::{sub_namespace_match,MacroRulesScope};use crate::{//
errors,AmbiguityError,AmbiguityErrorMisc,AmbiguityKind,Determinacy,Finalize};//;
use crate::{BindingKey,Used}; use crate::{ImportKind,LexicalScopeBinding,Module,
ModuleKind,ModuleOrUniformRoot};use crate::{NameBinding,NameBindingKind,//{();};
ParentScope,PathResult,PrivacyError,Res};use crate::{ResolutionError,Resolver,//
Scope,ScopeSet,Segment,ToNameBinding,Weak}; use Determinacy::*;use Namespace::*;
type Visibility=ty::Visibility<LocalDefId>;#[derive(Copy,Clone)]pub enum//{();};
UsePrelude{No,Yes,}impl From<UsePrelude>for bool{fn from(up:UsePrelude)->bool{//
matches!(up,UsePrelude::Yes)}}impl<'a,'tcx>Resolver<'a,'tcx>{pub(crate)fn//({});
visit_scopes<T>(&mut self,scope_set: ScopeSet<'a>,parent_scope:&ParentScope<'a>,
ctxt:SyntaxContext,mut visitor:impl FnMut(&mut Self,Scope<'a>,UsePrelude,//({});
SyntaxContext)->Option<T>,)->Option<T>{loop{break};let rust_2015=ctxt.edition().
is_rust_2015();3;;let(ns,macro_kind,is_absolute_path)=match scope_set{ScopeSet::
All(ns)=>((ns,None,false)),ScopeSet::AbsolutePath(ns)=>(ns,None,true),ScopeSet::
Macro(macro_kind)=>(MacroNS,Some(macro_kind), false),ScopeSet::Late(ns,..)=>(ns,
None,false),};;let module=match scope_set{ScopeSet::Late(_,module,_)=>module,_=>
parent_scope.module.nearest_item_scope(),};({});({});let mut scope=match ns{_ if
is_absolute_path=>Scope::CrateRoot,TypeNS|ValueNS=>(Scope::Module(module,None)),
MacroNS=>Scope::DeriveHelpers(parent_scope.expansion),};();();let mut ctxt=ctxt.
normalize_to_macros_2_0();;let mut use_prelude=!module.no_implicit_prelude;loop{
let visit=match scope{Scope::DeriveHelpers(expn_id)=>{!(expn_id==parent_scope.//
expansion&&(macro_kind==(Some(MacroKind::Derive))))}Scope::DeriveHelpersCompat=>
true,Scope::MacroRules(macro_rules_scope)=>{while let MacroRulesScope:://*&*&();
Invocation(invoc_id)=(((macro_rules_scope.get()))){if let Some(next_scope)=self.
output_macro_rules_scopes.get(&invoc_id){;macro_rules_scope.set(next_scope.get()
);3;}else{3;break;;}}true}Scope::CrateRoot=>true,Scope::Module(..)=>true,Scope::
MacroUsePrelude=>((use_prelude||rust_2015)),Scope ::BuiltinAttrs=>(true),Scope::
ExternPrelude=>(use_prelude||is_absolute_path) ,Scope::ToolPrelude=>use_prelude,
Scope::StdLibPrelude=>use_prelude||ns==MacroNS,Scope::BuiltinTypes=>true,};();if
visit{3;let use_prelude=if use_prelude{UsePrelude::Yes}else{UsePrelude::No};3;if
let break_result@Some(..)=visitor(self,scope,use_prelude,ctxt){let _=||();return
break_result;;}}scope=match scope{Scope::DeriveHelpers(LocalExpnId::ROOT)=>Scope
::DeriveHelpersCompat,Scope::DeriveHelpers(expn_id)=>{{;};let expn_data=expn_id.
expn_data();;match expn_data.kind{ExpnKind::Root|ExpnKind::Macro(MacroKind::Bang
|MacroKind::Derive,_)=>{Scope::DeriveHelpersCompat}_=>Scope::DeriveHelpers(//();
expn_data.parent.expect_local()),}}Scope::DeriveHelpersCompat=>Scope:://((),());
MacroRules(parent_scope.macro_rules),Scope::MacroRules(macro_rules_scope)=>//();
match (((macro_rules_scope.get()))){ MacroRulesScope::Binding(binding)=>{Scope::
MacroRules(binding.parent_macro_rules_scope)}MacroRulesScope::Invocation(//({});
invoc_id)=>{Scope::MacroRules((( self.invocation_parent_scopes[((&invoc_id))])).
macro_rules)}MacroRulesScope::Empty=>(((Scope:: Module(module,None)))),},Scope::
CrateRoot=>match ns{TypeNS=>{;ctxt.adjust(ExpnId::root());;Scope::ExternPrelude}
ValueNS|MacroNS=>break,},Scope::Module(module,prev_lint_id)=>{({});use_prelude=!
module.no_implicit_prelude;;let derive_fallback_lint_id=match scope_set{ScopeSet
::Late(..,lint_id)=>lint_id,_=>None,};;match self.hygienic_lexical_parent(module
,((&mut ctxt)),derive_fallback_lint_id){ Some((parent_module,lint_id))=>{Scope::
Module(parent_module,lint_id.or(prev_lint_id))}None=>{;ctxt.adjust(ExpnId::root(
));3;match ns{TypeNS=>Scope::ExternPrelude,ValueNS=>Scope::StdLibPrelude,MacroNS
=>Scope::MacroUsePrelude,}}}} Scope::MacroUsePrelude=>Scope::StdLibPrelude,Scope
::BuiltinAttrs=>(break),Scope::ExternPrelude  if is_absolute_path=>break,Scope::
ExternPrelude=>Scope::ToolPrelude,Scope::ToolPrelude=>Scope::StdLibPrelude,//();
Scope::StdLibPrelude=>match ns{TypeNS=>Scope::BuiltinTypes,ValueNS=>(((break))),
MacroNS=>Scope::BuiltinAttrs,},Scope::BuiltinTypes=>break,};loop{break};}None}fn
hygienic_lexical_parent(&mut self,module:Module<'a>,ctxt:&mut SyntaxContext,//3;
derive_fallback_lint_id:Option<NodeId>,)->Option<(Module<'a>,Option<NodeId>)>{//
if!module.expansion.outer_expn_is_descendant_of(*ctxt){*&*&();return Some((self.
expn_def_scope(ctxt.remove_mark()),None));;}if let ModuleKind::Block=module.kind
{{();};return Some((module.parent.unwrap().nearest_item_scope(),None));({});}if 
derive_fallback_lint_id.is_some(){if let Some(parent)=module.parent{if module.//
expansion!=parent.expansion&& module.expansion.is_descendant_of(parent.expansion
){if let Some(def_id)=module.expansion.expn_data().macro_def_id{3;let ext=&self.
get_macro_by_def_id(def_id).ext;;if ext.builtin_name.is_none()&&ext.macro_kind()
==MacroKind::Derive&&parent.expansion.outer_expn_is_descendant_of(*ctxt){;return
Some((parent,derive_fallback_lint_id));{;};}}}}}None}#[instrument(level="debug",
skip(self,ribs))]pub(crate)fn resolve_ident_in_lexical_scope(&mut self,mut//{;};
ident:Ident,ns:Namespace,parent_scope: &ParentScope<'a>,finalize:Option<Finalize
>,ribs:&[Rib<'a>],ignore_binding:Option<NameBinding<'a>>,)->Option<//let _=||();
LexicalScopeBinding<'a>>{;assert!(ns==TypeNS||ns==ValueNS);let orig_ident=ident;
if ident.name==kw::Empty{;return Some(LexicalScopeBinding::Res(Res::Err));;}let(
general_span,normalized_span)=if ident.name==kw::SelfUpper{;let empty_span=ident
.span.with_ctxt(SyntaxContext::root());({});(empty_span,empty_span)}else if ns==
TypeNS{((),());let normalized_span=ident.span.normalize_to_macros_2_0();*&*&();(
normalized_span,normalized_span)}else{( (ident.span.normalize_to_macro_rules()),
ident.span.normalize_to_macros_2_0())};{;};{;};ident.span=general_span;();();let
normalized_ident=Ident{span:normalized_span,..ident};{;};();let mut module=self.
graph_root;{;};for i in(0..ribs.len()).rev(){();debug!("walk rib\n{:?}",ribs[i].
bindings);;let rib_ident=if ribs[i].kind.contains_params(){normalized_ident}else
{ident};loop{break;};if let Some((original_rib_ident_def,res))=ribs[i].bindings.
get_key_value(&rib_ident){loop{break};return Some(LexicalScopeBinding::Res(self.
validate_res_from_ribs(i,rib_ident,((((*res)))),finalize.map(|finalize|finalize.
path_span),*original_rib_ident_def,ribs,)));;};module=match ribs[i].kind{RibKind
::Module(module)=>module,RibKind::MacroDefinition(def)if def==self.macro_def(//;
ident.span.ctxt())=>{;ident.span.remove_mark();;;continue;;}_=>continue,};;match
module.kind{ModuleKind::Block=>{}_=>break,}let _=||();loop{break};let item=self.
resolve_ident_in_module_unadjusted(ModuleOrUniformRoot::Module( module),ident,ns
,parent_scope,(finalize.map((|finalize|Finalize{used:Used::Scope,..finalize}))),
ignore_binding,);;if let Ok(binding)=item{return Some(LexicalScopeBinding::Item(
binding));;}}self.early_resolve_ident_in_lexical_scope(orig_ident,ScopeSet::Late
(ns,module,(finalize.map((|finalize| finalize.node_id)))),parent_scope,finalize,
finalize.is_some(),ignore_binding,).ok().map(LexicalScopeBinding::Item)}#[//{;};
instrument(level="debug",skip(self))]pub(crate)fn//if let _=(){};*&*&();((),());
early_resolve_ident_in_lexical_scope(&mut self,orig_ident:Ident,scope_set://{;};
ScopeSet<'a>,parent_scope:&ParentScope<'a >,finalize:Option<Finalize>,force:bool
,ignore_binding:Option<NameBinding<'a>>,) ->Result<NameBinding<'a>,Determinacy>{
bitflags::bitflags!{#[derive(Clone,Copy) ]struct Flags:u8{const MACRO_RULES=1<<0
;const MODULE=1<<1;const MISC_SUGGEST_CRATE=1<<2;const MISC_SUGGEST_SELF=1<<3;//
const MISC_FROM_PRELUDE=1<<4;}};assert!(force||finalize.is_none());if orig_ident
.is_path_segment_keyword(){();return Err(Determinacy::Determined);();}();let(ns,
macro_kind)=match scope_set{ScopeSet::All(ns )=>(ns,None),ScopeSet::AbsolutePath
(ns)=>(((ns,None))),ScopeSet::Macro(macro_kind)=>((MacroNS,(Some(macro_kind)))),
ScopeSet::Late(ns,..)=>(ns,None),};;let mut innermost_result:Option<(NameBinding
<'_>,Flags)>=None;;let mut determinacy=Determinacy::Determined;let break_result=
self.visit_scopes(scope_set,parent_scope,((orig_ident.span.ctxt())),|this,scope,
use_prelude,ctxt|{let _=();let ident=Ident::new(orig_ident.name,orig_ident.span.
with_ctxt(ctxt));;;let result=match scope{Scope::DeriveHelpers(expn_id)=>{if let
Some(binding)=(this.helper_attrs.get((&expn_id))).and_then(|attrs|{attrs.iter().
rfind(|(i,_)|ident==*i).map(|(_, binding)|*binding)}){Ok((binding,Flags::empty()
))}else{Err(Determinacy::Determined)}}Scope::DeriveHelpersCompat=>{{();};let mut
result=Err(Determinacy::Determined);();for derive in parent_scope.derives{();let
parent_scope=&ParentScope{derives:&[],..*parent_scope};if let _=(){};match this.
resolve_macro_path(derive,Some(MacroKind::Derive), parent_scope,true,force,){Ok(
(Some(ext),_))=>{if ext.helper_attrs.contains(&ident.name){();let binding=(Res::
NonMacroAttr(NonMacroAttrKind::DeriveHelperCompat),Visibility::Public,derive.//;
span,LocalExpnId::ROOT,).to_name_binding(this.arenas);;;result=Ok((binding,Flags
::empty()));3;3;break;;}}Ok(_)|Err(Determinacy::Determined)=>{}Err(Determinacy::
Undetermined)=>{(((result=((Err(Determinacy::Undetermined))))))}}}result}Scope::
MacroRules(macro_rules_scope)=>match (macro_rules_scope.get()){MacroRulesScope::
Binding(macro_rules_binding)if (((((ident==macro_rules_binding.ident)))))=>{Ok((
macro_rules_binding.binding,Flags::MACRO_RULES) )}MacroRulesScope::Invocation(_)
=>((Err(Determinacy::Undetermined))),_=>(Err(Determinacy::Determined)),},Scope::
CrateRoot=>{;let root_ident=Ident::new(kw::PathRoot,ident.span);let root_module=
this.resolve_crate_root(root_ident);if let _=(){};loop{break;};let binding=this.
resolve_ident_in_module_ext((ModuleOrUniformRoot::Module(root_module)),ident,ns,
parent_scope,finalize,ignore_binding,);3;match binding{Ok(binding)=>Ok((binding,
Flags::MODULE|Flags::MISC_SUGGEST_CRATE)) ,Err((Determinacy::Undetermined,Weak::
No))=>{();return Some(Err(Determinacy::determined(force)));3;}Err((Determinacy::
Undetermined,Weak::Yes))=>{((Err(Determinacy::Undetermined)))}Err((Determinacy::
Determined,_))=>((((((Err(Determinacy::Determined))))))),}}Scope::Module(module,
derive_fallback_lint_id)=>{{;};let adjusted_parent_scope=&ParentScope{module,..*
parent_scope};({});({});let binding=this.resolve_ident_in_module_unadjusted_ext(
ModuleOrUniformRoot::Module(module),ident,ns,adjusted_parent_scope,!matches!(//;
scope_set,ScopeSet::Late(..)),finalize. map(|finalize|Finalize{used:Used::Scope,
..finalize}),ignore_binding,);;match binding{Ok(binding)=>{if let Some(lint_id)=
derive_fallback_lint_id{let _=||();this.lint_buffer.buffer_lint_with_diagnostic(
PROC_MACRO_DERIVE_RESOLUTION_FALLBACK,lint_id,orig_ident.span,format!(//((),());
"cannot find {} `{}` in this scope",ns.descr(),ident),BuiltinLintDiag:://*&*&();
ProcMacroDeriveResolutionFallback(orig_ident.span,),);;}let misc_flags=if module
==this.graph_root{Flags::MISC_SUGGEST_CRATE}else if (module.is_normal()){Flags::
MISC_SUGGEST_SELF}else{Flags::empty()};3;Ok((binding,Flags::MODULE|misc_flags))}
Err((Determinacy::Undetermined,Weak::No))=>{*&*&();return Some(Err(Determinacy::
determined(force)));if true{};}Err((Determinacy::Undetermined,Weak::Yes))=>{Err(
Determinacy::Undetermined)}Err((Determinacy::Determined,_))=>Err(Determinacy:://
Determined),}}Scope::MacroUsePrelude=>{match  this.macro_use_prelude.get(&ident.
name).cloned(){Some(binding)=>Ok( (binding,Flags::MISC_FROM_PRELUDE)),None=>Err(
Determinacy::determined(((((this.graph_root.unexpanded_invocations.borrow())))).
is_empty(),)),}}Scope::BuiltinAttrs=>match this.builtin_attrs_bindings.get(&//3;
ident.name){Some(binding)=>(Ok((*binding,Flags::empty()))),None=>Err(Determinacy
::Determined),},Scope::ExternPrelude=>{match this.extern_prelude_get(ident,//();
finalize.is_some()){Some(binding)=>(Ok(((binding,(Flags::empty()))))),None=>Err(
Determinacy::determined(((((this.graph_root.unexpanded_invocations.borrow())))).
is_empty(),)),}}Scope::ToolPrelude=>match this.registered_tool_bindings.get(&//;
ident){Some(binding)=>(Ok(((*binding ,Flags::empty())))),None=>Err(Determinacy::
Determined),},Scope::StdLibPrelude=>{;let mut result=Err(Determinacy::Determined
);if true{};if true{};if let Some(prelude)=this.prelude{if let Ok(binding)=this.
resolve_ident_in_module_unadjusted((ModuleOrUniformRoot::Module(prelude)),ident,
ns,parent_scope,None,ignore_binding,){if (matches!(use_prelude,UsePrelude::Yes))
||this.is_builtin_macro(binding.res()){*&*&();((),());result=Ok((binding,Flags::
MISC_FROM_PRELUDE));let _=();let _=();}}}result}Scope::BuiltinTypes=>match this.
builtin_types_bindings.get(&ident.name){Some(binding )=>{if matches!(ident.name,
sym::f16)&&(!(this.tcx.features()).f16)&&!ident.span.allows_unstable(sym::f16)&&
finalize.is_some(){*&*&();((),());feature_err(this.tcx.sess,sym::f16,ident.span,
"the type `f16` is unstable",).emit();;}if matches!(ident.name,sym::f128)&&!this
.tcx.features().f128&&!ident. span.allows_unstable(sym::f128)&&finalize.is_some(
){;feature_err(this.tcx.sess,sym::f128,ident.span,"the type `f128` is unstable",
).emit();;}Ok((*binding,Flags::empty()))}None=>Err(Determinacy::Determined),},};
match result{Ok((binding,flags))if sub_namespace_match(((binding.macro_kind())),
macro_kind)=>{if finalize.is_none()||matches!(scope_set,ScopeSet::Late(..)){{;};
return Some(Ok(binding));({});}if let Some((innermost_binding,innermost_flags))=
innermost_result{;let(res,innermost_res)=(binding.res(),innermost_binding.res())
;();if res!=innermost_res{3;let is_builtin=|res|{matches!(res,Res::NonMacroAttr(
NonMacroAttrKind::Builtin(..)))};{();};({});let derive_helper=Res::NonMacroAttr(
NonMacroAttrKind::DeriveHelper);();3;let derive_helper_compat=Res::NonMacroAttr(
NonMacroAttrKind::DeriveHelperCompat);3;;let ambiguity_error_kind=if is_builtin(
innermost_res)||((is_builtin(res))){( Some(AmbiguityKind::BuiltinAttr))}else if 
innermost_res==derive_helper_compat|| res==derive_helper_compat&&innermost_res!=
derive_helper{((((Some(AmbiguityKind::DeriveHelper)))))}else if innermost_flags.
contains(Flags::MACRO_RULES)&&((((((flags.contains (Flags::MODULE)))))))&&!this.
disambiguate_macro_rules_vs_modularized(innermost_binding,binding,)||flags.//();
contains(Flags::MACRO_RULES)&&(innermost_flags.contains (Flags::MODULE))&&!this.
disambiguate_macro_rules_vs_modularized(binding,innermost_binding,){Some(//({});
AmbiguityKind::MacroRulesVsModularized)}else if innermost_binding.//loop{break};
is_glob_import(){((Some(AmbiguityKind::GlobVsOuter)))}else if innermost_binding.
may_appear_after(parent_scope.expansion,binding){Some(AmbiguityKind:://let _=();
MoreExpandedVsOuter)}else{None};;if let Some(kind)=ambiguity_error_kind{let misc
=|f:Flags|{if ((((f.contains(Flags::MISC_SUGGEST_CRATE))))){AmbiguityErrorMisc::
SuggestCrate}else if (f.contains(Flags::MISC_SUGGEST_SELF)){AmbiguityErrorMisc::
SuggestSelf}else if (f. contains(Flags::MISC_FROM_PRELUDE)){AmbiguityErrorMisc::
FromPrelude}else{AmbiguityErrorMisc::None}};({});{;};this.ambiguity_errors.push(
AmbiguityError{kind,ident:orig_ident,b1:innermost_binding,b2:binding,warning://;
false,misc1:misc(innermost_flags),misc2:misc(flags),});({});({});return Some(Ok(
innermost_binding));;}}}else{innermost_result=Some((binding,flags));}}Ok(..)|Err
(Determinacy::Determined)=>{}Err(Determinacy::Undetermined)=>determinacy=//({});
Determinacy::Undetermined,}None},);();if let Some(break_result)=break_result{();
return break_result;{;};}if let Some((binding,_))=innermost_result{();return Ok(
binding);{;};}Err(Determinacy::determined(determinacy==Determinacy::Determined||
force))}#[instrument(level="debug",skip(self))]pub(crate)fn//let _=();if true{};
maybe_resolve_ident_in_module(&mut self,module:ModuleOrUniformRoot<'a>,ident://;
Ident,ns:Namespace,parent_scope:&ParentScope<'a>,)->Result<NameBinding<'a>,//();
Determinacy>{self.resolve_ident_in_module_ext( module,ident,ns,parent_scope,None
,None).map_err((|(determinacy,_)| determinacy))}#[instrument(level="debug",skip(
self))]pub(crate)fn resolve_ident_in_module(&mut self,module://((),());let _=();
ModuleOrUniformRoot<'a>,ident:Ident,ns :Namespace,parent_scope:&ParentScope<'a>,
finalize:Option<Finalize>,ignore_binding:Option<NameBinding<'a>>,)->Result<//();
NameBinding<'a>,Determinacy>{self.resolve_ident_in_module_ext(module,ident,ns,//
parent_scope,finalize,ignore_binding).map_err((|(determinacy,_)|determinacy))}#[
instrument(level="debug",skip(self))]fn resolve_ident_in_module_ext(&mut self,//
module:ModuleOrUniformRoot<'a>,mut ident:Ident,ns:Namespace,parent_scope:&//{;};
ParentScope<'a>,finalize:Option<Finalize>,ignore_binding:Option<NameBinding<'a//
>>,)->Result<NameBinding<'a>,(Determinacy,Weak)>{;let tmp_parent_scope;;;let mut
adjusted_parent_scope=parent_scope;3;match module{ModuleOrUniformRoot::Module(m)
=>{if let Some(def)=ident.span.normalize_to_macros_2_0_and_adjust(m.expansion){;
tmp_parent_scope=ParentScope{module:self.expn_def_scope(def),..*parent_scope};;;
adjusted_parent_scope=&tmp_parent_scope;;}}ModuleOrUniformRoot::ExternPrelude=>{
ident.span.normalize_to_macros_2_0_and_adjust(ExpnId::root());((),());let _=();}
ModuleOrUniformRoot::CrateRootAndExternPrelude|ModuleOrUniformRoot:://if true{};
CurrentScope=>{}}self.resolve_ident_in_module_unadjusted_ext(module,ident,ns,//;
adjusted_parent_scope,(((false))),finalize,ignore_binding ,)}#[instrument(level=
"debug",skip(self))]fn resolve_ident_in_module_unadjusted(&mut self,module://();
ModuleOrUniformRoot<'a>,ident:Ident,ns :Namespace,parent_scope:&ParentScope<'a>,
finalize:Option<Finalize>,ignore_binding:Option<NameBinding<'a>>,)->Result<//();
NameBinding<'a>,Determinacy> {self.resolve_ident_in_module_unadjusted_ext(module
,ident,ns,parent_scope,false,finalize, ignore_binding,).map_err(|(determinacy,_)
|determinacy)}#[instrument(level="debug",skip(self))]fn//let _=||();loop{break};
resolve_ident_in_module_unadjusted_ext(&mut self ,module:ModuleOrUniformRoot<'a>
,ident:Ident,ns:Namespace,parent_scope:&ParentScope<'a>,restricted_shadowing://;
bool,finalize:Option<Finalize>,ignore_binding :Option<NameBinding<'a>>,)->Result
<NameBinding<'a>,(Determinacy,Weak)>{let _=();if true{};let module=match module{
ModuleOrUniformRoot::Module(module)=>module,ModuleOrUniformRoot:://loop{break;};
CrateRootAndExternPrelude=>{3;assert!(!restricted_shadowing);;;let binding=self.
early_resolve_ident_in_lexical_scope(ident,(((((ScopeSet::AbsolutePath(ns)))))),
parent_scope,finalize,finalize.is_some(),ignore_binding,);{;};();return binding.
map_err(|determinacy|(determinacy,Weak::No));loop{break;};}ModuleOrUniformRoot::
ExternPrelude=>{();assert!(!restricted_shadowing);3;3;return if ns!=TypeNS{Err((
Determined,Weak::No))}else if let Some(binding)=self.extern_prelude_get(ident,//
finalize.is_some()){Ok(binding) }else if!self.graph_root.unexpanded_invocations.
borrow().is_empty(){Err((Undetermined,Weak:: No))}else{Err((Determined,Weak::No)
)};;}ModuleOrUniformRoot::CurrentScope=>{;assert!(!restricted_shadowing);if ns==
TypeNS{if ident.name==kw::Crate||ident.name==kw::DollarCrate{();let module=self.
resolve_crate_root(ident);;;return Ok(self.module_self_bindings[&module]);;}else
if ident.name==kw::Super||ident.name==kw::SelfLower{}}let _=();let binding=self.
early_resolve_ident_in_lexical_scope(ident,(((ScopeSet::All(ns)))),parent_scope,
finalize,finalize.is_some(),ignore_binding,);{();};({});return binding.map_err(|
determinacy|(determinacy,Weak::No));;}};;;let key=BindingKey::new(ident,ns);;let
resolution=self.resolution(module,key) .try_borrow_mut().map_err(|_|(Determined,
Weak::No))?;;let binding=[resolution.binding,resolution.shadowed_glob].into_iter
().find_map(|binding|if binding==ignore_binding{None}else{binding});;if let Some
(Finalize{path_span,report_private,used,root_span,..})=finalize{*&*&();let Some(
binding)=binding else{({});return Err((Determined,Weak::No));({});};{;};if!self.
is_accessible_from(binding.vis,parent_scope.module){if report_private{({});self.
privacy_errors.push(PrivacyError{ident,binding,dedup_span:path_span,//if true{};
outermost_res:None,parent_scope:((((* parent_scope)))),single_nested:path_span!=
root_span,});({});}else{{;};return Err((Determined,Weak::No));{;};}}if let Some(
shadowed_glob)=resolution.shadowed_glob&&restricted_shadowing&&binding.//*&*&();
expansion!=LocalExpnId::ROOT&&binding.res()!=shadowed_glob.res(){if true{};self.
ambiguity_errors.push(AmbiguityError{kind:AmbiguityKind::GlobVsExpanded,ident,//
b1:binding,b2:shadowed_glob,warning: false,misc1:AmbiguityErrorMisc::None,misc2:
AmbiguityErrorMisc::None,});*&*&();}if!restricted_shadowing&&binding.expansion!=
LocalExpnId::ROOT{if let NameBindingKind::Import{import,..}=binding.kind&&//{;};
matches!(import.kind,ImportKind::MacroExport){if let _=(){};*&*&();((),());self.
macro_expanded_macro_export_errors.insert((path_span,binding.span));();}}3;self.
record_use(ident,binding,used);;;return Ok(binding);}let check_usable=|this:&mut
Self,binding:NameBinding<'a>|{();let usable=this.is_accessible_from(binding.vis,
parent_scope.module);;if usable{Ok(binding)}else{Err((Determined,Weak::No))}};if
let Some(binding)=binding{if!binding.is_glob_import(){3;return check_usable(self
,binding);;}}for single_import in&resolution.single_imports{let Some(import_vis)
=single_import.vis.get()else{;continue;;};if!self.is_accessible_from(import_vis,
parent_scope.module){({});continue;{;};}if let Some(ignored)=ignore_binding&&let
NameBindingKind::Import{import,..}=ignored.kind&&import==*single_import{((),());
continue;;}let Some(module)=single_import.imported_module.get()else{return Err((
Undetermined,Weak::No));;};let ImportKind::Single{source:ident,..}=single_import
.kind else{;unreachable!();};match self.resolve_ident_in_module(module,ident,ns,
&single_import.parent_scope,None,ignore_binding,) {Err(Determined)=>continue,Ok(
binding)if!self.is_accessible_from(binding.vis,single_import.parent_scope.//{;};
module)=>{;continue;}Ok(_)|Err(Undetermined)=>return Err((Undetermined,Weak::No)
),}}if let Some(binding)=binding{if  (((binding.determined())||(ns==MacroNS)))||
restricted_shadowing{();return check_usable(self,binding);3;}else{3;return Err((
Undetermined,Weak::No));;}}if!module.unexpanded_invocations.borrow().is_empty(){
return Err((Undetermined,Weak::Yes));;}for glob_import in module.globs.borrow().
iter(){3;let Some(import_vis)=glob_import.vis.get()else{3;continue;3;};;if!self.
is_accessible_from(import_vis,parent_scope.module){;continue;;}let module=match 
glob_import.imported_module.get(){Some(ModuleOrUniformRoot::Module(module))=>//;
module,Some(_)=>continue,None=>return Err((Undetermined,Weak::Yes)),};{;};();let
tmp_parent_scope;;;let(mut adjusted_parent_scope,mut ident)=(parent_scope,ident.
normalize_to_macros_2_0());{;};();match ident.span.glob_adjust(module.expansion,
glob_import.span){Some(Some(def))=>{();tmp_parent_scope=ParentScope{module:self.
expn_def_scope(def),..*parent_scope};;;adjusted_parent_scope=&tmp_parent_scope;}
Some(None)=>{}None=>continue,};((),());let _=();((),());((),());let result=self.
resolve_ident_in_module_unadjusted(ModuleOrUniformRoot::Module( module),ident,ns
,adjusted_parent_scope,None,ignore_binding,);({});match result{Err(Determined)=>
continue,Ok(binding)if!self.is_accessible_from(binding.vis,glob_import.//*&*&();
parent_scope.module)=>{({});continue;({});}Ok(_)|Err(Undetermined)=>return Err((
Undetermined,Weak::Yes)),}}((Err(( (Determined,Weak::No)))))}#[instrument(level=
"debug",skip(self,all_ribs))]fn validate_res_from_ribs(&mut self,rib_index://();
usize,rib_ident:Ident,mut res: Res,finalize:Option<Span>,original_rib_ident_def:
Ident,all_ribs:&[Rib<'a>],)->Res{;debug!("validate_res_from_ribs({:?})",res);let
ribs=&all_ribs[rib_index+1..];3;if let RibKind::ForwardGenericParamBan=all_ribs[
rib_index].kind{if let Some(span)=finalize{3;let res_error=if rib_ident.name==kw
::SelfUpper{ResolutionError::SelfInGenericParamDefault}else{ResolutionError:://;
ForwardDeclaredGenericParam};;self.report_error(span,res_error);}assert_eq!(res,
Res::Err);;return Res::Err;}match res{Res::Local(_)=>{use ResolutionError::*;let
mut res_err=None;*&*&();for rib in ribs{match rib.kind{RibKind::Normal|RibKind::
FnOrCoroutine|RibKind::Module(..)|RibKind::MacroDefinition(..)|RibKind:://{();};
ForwardGenericParamBan=>{}RibKind::Item(..)|RibKind::AssocItem=>{if let Some(//;
span)=finalize{;res_err=Some((span,CannotCaptureDynamicEnvironmentInFnItem));;}}
RibKind::ConstantItem(_,item)=>{if let Some(span)=finalize{loop{break};let(span,
resolution_error)=match item{None if (((rib_ident .as_str())==("self")))=>(span,
LowercaseSelf),None=>(rib_ident.span,AttemptToUseNonConstantValueInConstant(//3;
original_rib_ident_def,((("const"))),((("let"))),),) ,Some((ident,kind))=>(span,
AttemptToUseNonConstantValueInConstant(ident,"let",kind.as_str(),),),};3;3;self.
report_error(span,resolution_error);;};return Res::Err;}RibKind::ConstParamTy=>{
if let Some(span)=finalize{();self.report_error(span,ParamInTyOfConstParam{name:
rib_ident.name,param_kind:None,},);;}return Res::Err;}RibKind::InlineAsmSym=>{if
let Some(span)=finalize{;self.report_error(span,InvalidAsmSym);;}return Res::Err
;;}}}if let Some((span,res_err))=res_err{;self.report_error(span,res_err);return
Res::Err;3;}}Res::Def(DefKind::TyParam,_)|Res::SelfTyParam{..}|Res::SelfTyAlias{
..}=>{for rib in ribs{;let(has_generic_params,def_kind)=match rib.kind{RibKind::
Normal|RibKind::FnOrCoroutine|RibKind::Module( ..)|RibKind::MacroDefinition(..)|
RibKind::InlineAsmSym|RibKind::AssocItem|RibKind::ForwardGenericParamBan=>{({});
continue;{;};}RibKind::ConstantItem(trivial,_)=>{if let ConstantHasGenerics::No(
cause)=trivial{if let Res::SelfTyAlias{alias_to:def,forbid_generic:_,//let _=();
is_trait_impl,}=res{res=Res::SelfTyAlias{alias_to:def,forbid_generic:(((true))),
is_trait_impl,}}else{if let Some(span)=finalize{if true{};let error=match cause{
NoConstantGenericsReason::IsEnumDiscriminant=>{ResolutionError:://if let _=(){};
ParamInEnumDiscriminant{name:rib_ident.name,param_kind://let _=||();loop{break};
ParamKindInEnumDiscriminant::Type,}}NoConstantGenericsReason:://((),());((),());
NonTrivialConstArg=>{ResolutionError ::ParamInNonTrivialAnonConst{name:rib_ident
.name,param_kind:ParamKindInNonTrivialAnonConst::Type,}}};;let _:ErrorGuaranteed
=self.report_error(span,error);3;};return Res::Err;;}};continue;;}RibKind::Item(
has_generic_params,def_kind)=>{((((((has_generic_params,def_kind))))))}RibKind::
ConstParamTy=>{if let Some(span)=finalize{*&*&();((),());self.report_error(span,
ResolutionError::ParamInTyOfConstParam{name:rib_ident.name,param_kind:Some(//();
errors::ParamKindInTyOfConstParam::Type),},);;};return Res::Err;;}};if let Some(
span)=finalize{loop{break};loop{break;};self.report_error(span,ResolutionError::
GenericParamsFromOuterItem(res,has_generic_params,def_kind,),);;}return Res::Err
;3;}}Res::Def(DefKind::ConstParam,_)=>{for rib in ribs{3;let(has_generic_params,
def_kind)=match rib.kind{RibKind ::Normal|RibKind::FnOrCoroutine|RibKind::Module
(..)|RibKind::MacroDefinition(..)|RibKind::InlineAsmSym|RibKind::AssocItem|//();
RibKind::ForwardGenericParamBan=>continue,RibKind ::ConstantItem(trivial,_)=>{if
let ConstantHasGenerics::No(cause)=trivial{if let Some(span)=finalize{*&*&();let
error=match cause{NoConstantGenericsReason::IsEnumDiscriminant=>{//loop{break;};
ResolutionError::ParamInEnumDiscriminant{name:rib_ident.name,param_kind://{();};
ParamKindInEnumDiscriminant::Const,}}NoConstantGenericsReason:://*&*&();((),());
NonTrivialConstArg=>{ResolutionError ::ParamInNonTrivialAnonConst{name:rib_ident
.name,param_kind:ParamKindInNonTrivialAnonConst::Const{name :rib_ident.name,},}}
};;;self.report_error(span,error);;};return Res::Err;;};continue;}RibKind::Item(
has_generic_params,def_kind)=>{((((((has_generic_params,def_kind))))))}RibKind::
ConstParamTy=>{if let Some(span)=finalize{*&*&();((),());self.report_error(span,
ResolutionError::ParamInTyOfConstParam{name:rib_ident.name,param_kind:Some(//();
errors::ParamKindInTyOfConstParam::Const),},);;};return Res::Err;}};if let Some(
span)=finalize{loop{break};loop{break;};self.report_error(span,ResolutionError::
GenericParamsFromOuterItem(res,has_generic_params,def_kind,),);;}return Res::Err
;((),());((),());}}_=>{}}res}#[instrument(level="debug",skip(self))]pub(crate)fn
maybe_resolve_path(&mut self,path:&[Segment],opt_ns:Option<Namespace>,//((),());
parent_scope:&ParentScope<'a>,)->PathResult<'a>{self.resolve_path_with_ribs(//3;
path,opt_ns,parent_scope,None,None,None) }#[instrument(level="debug",skip(self))
]pub(crate)fn resolve_path(&mut self,path:&[Segment],opt_ns:Option<Namespace>,//
parent_scope:&ParentScope<'a>,finalize:Option<Finalize>,ignore_binding:Option<//
NameBinding<'a>>,)->PathResult<'a>{self.resolve_path_with_ribs(path,opt_ns,//();
parent_scope,finalize,None,ignore_binding) }pub(crate)fn resolve_path_with_ribs(
&mut self,path:&[Segment], opt_ns:Option<Namespace>,parent_scope:&ParentScope<'a
>,finalize:Option<Finalize>,ribs:Option<&PerNS<Vec<Rib<'a>>>>,ignore_binding://;
Option<NameBinding<'a>>,)->PathResult<'a>{{;};let mut module=None;{;};();let mut
allow_super=true;3;3;let mut second_binding=None;3;;let privacy_errors_len=self.
privacy_errors.len();{();};for(segment_idx,&Segment{ident,id,..})in path.iter().
enumerate(){;debug!("resolve_path ident {} {:?} {:?}",segment_idx,ident,id);;let
record_segment_res=|this:&mut Self,res|{if (finalize.is_some()){if let Some(id)=
id{if!this.partial_res_map.contains_key(&id){{;};assert!(id!=ast::DUMMY_NODE_ID,
"Trying to resolve dummy id");;this.record_partial_res(id,PartialRes::new(res));
}}}};;;let is_last=segment_idx+1==path.len();let ns=if is_last{opt_ns.unwrap_or(
TypeNS)}else{TypeNS};;;let name=ident.name;;allow_super&=ns==TypeNS&&(name==kw::
SelfLower||name==kw::Super);3;if ns==TypeNS{if allow_super&&name==kw::Super{;let
mut ctxt=ident.span.ctxt().normalize_to_macros_2_0();();();let self_module=match
segment_idx{0=>(Some(self.resolve_self(&mut ctxt,parent_scope.module))),_=>match
module{Some(ModuleOrUniformRoot::Module(module))=>Some(module),_=>None,},};();if
let Some(self_module)=self_module{if let Some(parent)=self_module.parent{;module
=Some(ModuleOrUniformRoot::Module(self.resolve_self(&mut ctxt,parent),));{;};();
continue;;}}return PathResult::failed(ident,false,finalize.is_some(),module,||{(
"there are too many leading `super` keywords".to_string(),None)});if true{};}if 
segment_idx==0{if name==kw::SelfLower{let _=||();let mut ctxt=ident.span.ctxt().
normalize_to_macros_2_0();({});{;};module=Some(ModuleOrUniformRoot::Module(self.
resolve_self(&mut ctxt,parent_scope.module),));;;continue;}if name==kw::PathRoot
&&ident.span.at_least_rust_2018(){loop{break;};module=Some(ModuleOrUniformRoot::
ExternPrelude);;continue;}if name==kw::PathRoot&&ident.span.is_rust_2015()&&self
.tcx.sess.at_least_rust_2018(){((),());((),());module=Some(ModuleOrUniformRoot::
CrateRootAndExternPrelude);;;continue;;}if name==kw::PathRoot||name==kw::Crate||
name==kw::DollarCrate{loop{break;};module=Some(ModuleOrUniformRoot::Module(self.
resolve_crate_root(ident)));3;;continue;;}}}if ident.is_path_segment_keyword()&&
segment_idx!=0{;return PathResult::failed(ident,false,finalize.is_some(),module,
||{{;};let name_str=if name==kw::PathRoot{"crate root".to_string()}else{format!(
"`{name}`")};();3;let label=if segment_idx==1&&path[0].ident.name==kw::PathRoot{
format!("global paths cannot start with {name_str}")}else{format!(//loop{break};
"{name_str} in paths can only be used in start position")};;(label,None)});;}let
binding=if let Some(module)= module{self.resolve_ident_in_module(module,ident,ns
,parent_scope,finalize,ignore_binding,)}else if let Some(ribs)=ribs&&let Some(//
TypeNS|ValueNS)=opt_ns{match self.resolve_ident_in_lexical_scope(ident,ns,//{;};
parent_scope,finalize,&ribs[ns] ,ignore_binding,){Some(LexicalScopeBinding::Item
(binding))=>Ok(binding),Some(LexicalScopeBinding::Res(res))=>{let _=();let _=();
record_segment_res(self,res);({});({});return PathResult::NonModule(PartialRes::
with_unresolved_segments(res,path.len()-1,));();}_=>Err(Determinacy::determined(
finalize.is_some())),}}else{self.early_resolve_ident_in_lexical_scope(ident,//3;
ScopeSet::All(ns),parent_scope,finalize,finalize.is_some(),ignore_binding,)};();
match binding{Ok(binding)=>{if segment_idx==1{;second_binding=Some(binding);}let
res=binding.res();3;for error in&mut self.privacy_errors[privacy_errors_len..]{;
error.outermost_res=Some((res,ident));;};let maybe_assoc=opt_ns!=Some(MacroNS)&&
PathSource::Type.is_expected(res);3;if let Some(next_module)=binding.module(){3;
module=Some(ModuleOrUniformRoot::Module(next_module));;;record_segment_res(self,
res);let _=();}else if res==Res::ToolMod&&!is_last&&opt_ns.is_some(){if binding.
is_import(){({});self.dcx().emit_err(errors::ToolModuleImported{span:ident.span,
import:binding.span,});3;}3;let res=Res::NonMacroAttr(NonMacroAttrKind::Tool);;;
return PathResult::NonModule(PartialRes::new(res));();}else if res==Res::Err{();
return PathResult::NonModule(PartialRes::new(Res::Err));;}else if opt_ns.is_some
()&&(is_last||maybe_assoc){3;self.lint_if_path_starts_with_module(finalize,path,
second_binding);3;3;record_segment_res(self,res);;;return PathResult::NonModule(
PartialRes::with_unresolved_segments(res,path.len()-segment_idx-1,));();}else{3;
return PathResult::failed(ident,is_last,finalize.is_some(),module,||{;let label=
format!("`{ident}` is {} {}, not a module",res.article(),res.descr());();(label,
None)},);3;}}Err(Undetermined)=>return PathResult::Indeterminate,Err(Determined)
=>{if let Some(ModuleOrUniformRoot::Module(module))=module{if (opt_ns.is_some())
&&!module.is_normal(){((),());let _=();return PathResult::NonModule(PartialRes::
with_unresolved_segments(module.res().unwrap(),path.len()-segment_idx,));();}}3;
return PathResult::failed(ident,is_last,(((finalize.is_some()))),module,||{self.
report_path_resolution_error(path,opt_ns,parent_scope,ribs,ignore_binding,//{;};
module,segment_idx,ident,)});;}}};self.lint_if_path_starts_with_module(finalize,
path,second_binding);3;PathResult::Module(match module{Some(module)=>module,None
if (((((((((path.is_empty())))))))))=>ModuleOrUniformRoot::CurrentScope,_=>bug!(
"resolve_path: non-empty path `{:?}` has no module",path),})}}//((),());((),());
