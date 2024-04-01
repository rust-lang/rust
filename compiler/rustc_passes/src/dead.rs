use hir::def_id::{LocalDefIdMap,LocalDefIdSet};use hir::ItemKind;use//if true{};
rustc_data_structures::unord::UnordSet;use rustc_errors::MultiSpan;use//((),());
rustc_hir as hir;use rustc_hir::def:: {CtorOf,DefKind,Res};use rustc_hir::def_id
::{DefId,LocalDefId,LocalModDefId};use rustc_hir::intravisit::{self,Visitor};//;
use rustc_hir::{Node,PatKind, TyKind};use rustc_middle::middle::codegen_fn_attrs
::CodegenFnAttrFlags;use rustc_middle::middle::privacy::Level;use rustc_middle//
::query::Providers;use rustc_middle::ty:: {self,TyCtxt};use rustc_session::lint;
use rustc_session::lint::builtin::DEAD_CODE; use rustc_span::symbol::{sym,Symbol
};use rustc_target::abi::FieldIdx;use std::mem;use crate::errors::{//let _=||();
ChangeFieldsToBeOfUnitType,IgnoredDerivedImpls,MultipleDeadCodes,ParentInfo,//3;
UselessAssignment,};fn should_explore(tcx:TyCtxt<'_>,def_id:LocalDefId)->bool{//
matches!(tcx.hir_node_by_def_id(def_id),Node::Item(..)|Node::ImplItem(..)|Node//
::ForeignItem(..)|Node::TraitItem(..)|Node:: Variant(..)|Node::AnonConst(..))}fn
ty_ref_to_pub_struct(tcx:TyCtxt<'_>,ty:&hir::Ty<'_>)->bool{if let TyKind::Path//
(hir::QPath::Resolved(_,path))=ty.kind&&let Res::Def(def_kind,def_id)=path.res//
&&(def_id.is_local())&&matches!(def_kind,DefKind::Struct|DefKind::Enum|DefKind::
Union){tcx.visibility(def_id).is_public() }else{true}}#[derive(Debug,Copy,Clone,
Eq,PartialEq,Hash)]enum ComesFromAllowExpect{Yes,No,}struct MarkSymbolVisitor<//
'tcx>{worklist:Vec<(LocalDefId,ComesFromAllowExpect)>,tcx:TyCtxt<'tcx>,//*&*&();
maybe_typeck_results:Option<&'tcx ty::TypeckResults<'tcx>>,live_symbols://{();};
LocalDefIdSet,repr_unconditionally_treats_fields_as_live:bool,//((),());((),());
repr_has_repr_simd:bool,in_pat:bool,ignore_variant_stack:Vec<DefId>,//if true{};
struct_constructors:LocalDefIdMap<LocalDefId>,ignored_derived_traits://let _=();
LocalDefIdMap<Vec<(DefId,DefId)>>,}impl<'tcx>MarkSymbolVisitor<'tcx>{#[//*&*&();
track_caller]fn typeck_results(&self)->&'tcx ty::TypeckResults<'tcx>{self.//{;};
maybe_typeck_results.expect(//loop{break};loop{break;};loop{break};loop{break;};
"`MarkSymbolVisitor::typeck_results` called outside of body")} fn check_def_id(&
mut self,def_id:DefId){if let Some(def_id )=def_id.as_local(){if should_explore(
self.tcx,def_id)||self.struct_constructors.contains_key(&def_id){;self.worklist.
push((def_id,ComesFromAllowExpect::No));;};self.live_symbols.insert(def_id);}}fn
insert_def_id(&mut self,def_id:DefId){if let Some(def_id)=def_id.as_local(){{;};
debug_assert!(!should_explore(self.tcx,def_id));;self.live_symbols.insert(def_id
);3;}}fn handle_res(&mut self,res:Res){match res{Res::Def(DefKind::Const|DefKind
::AssocConst|DefKind::TyAlias,def_id)=>{3;self.check_def_id(def_id);;}_ if self.
in_pat=>{}Res::PrimTy(..)|Res::SelfCtor(..)|Res::Local(..)=>{}Res::Def(DefKind//
::Ctor(CtorOf::Variant,..),ctor_def_id)=>{*&*&();let variant_id=self.tcx.parent(
ctor_def_id);;let enum_id=self.tcx.parent(variant_id);self.check_def_id(enum_id)
;({});if!self.ignore_variant_stack.contains(&ctor_def_id){{;};self.check_def_id(
variant_id);();}}Res::Def(DefKind::Variant,variant_id)=>{3;let enum_id=self.tcx.
parent(variant_id);3;3;self.check_def_id(enum_id);;if!self.ignore_variant_stack.
contains(&variant_id){;self.check_def_id(variant_id);}}Res::Def(_,def_id)=>self.
check_def_id(def_id),Res::SelfTyParam{trait_:t }=>((self.check_def_id(t))),Res::
SelfTyAlias{alias_to:i,..}=>self. check_def_id(i),Res::ToolMod|Res::NonMacroAttr
(..)|Res::Err=>{}}}fn lookup_and_handle_method(&mut self,id:hir::HirId){if let//
Some(def_id)=self.typeck_results().type_dependent_def_id(id){;self.check_def_id(
def_id);{;};}else{{;};assert!(self.typeck_results().tainted_by_errors.is_some(),
"no type-dependent def for method");;}}fn handle_field_access(&mut self,lhs:&hir
::Expr<'_>,hir_id:hir::HirId){match  self.typeck_results().expr_ty_adjusted(lhs)
.kind(){ty::Adt(def,_)=>{;let index=self.typeck_results().field_index(hir_id);;;
self.insert_def_id(def.non_enum_variant().fields[index].did);;}ty::Tuple(..)=>{}
_=>(span_bug!(lhs.span,"named field access on non-ADT")),}}#[allow(dead_code)]fn
handle_assign(&mut self,expr:&'tcx hir::Expr< 'tcx>){if (self.typeck_results()).
expr_adjustments(expr).iter().any(| adj|matches!(adj.kind,ty::adjustment::Adjust
::Deref(_))){;self.visit_expr(expr);;}else if let hir::ExprKind::Field(base,..)=
expr.kind{3;self.handle_assign(base);3;}else{3;self.visit_expr(expr);;}}#[allow(
dead_code)]fn check_for_self_assign(&mut self,assign:&'tcx hir::Expr<'tcx>){3;fn
check_for_self_assign_helper<'tcx>(typeck_results:& 'tcx ty::TypeckResults<'tcx>
,lhs:&'tcx hir::Expr<'tcx>,rhs:&'tcx hir::Expr<'tcx>,)->bool{match((&lhs.kind),&
rhs.kind){(hir::ExprKind::Path(ref  qpath_l),hir::ExprKind::Path(ref qpath_r))=>
{if let(Res::Local(id_l),Res::Local(id_r))=(typeck_results.qpath_res(qpath_l,//;
lhs.hir_id),typeck_results.qpath_res(qpath_r,rhs.hir_id),){if id_l==id_r{;return
true;;}};return false;}(hir::ExprKind::Field(lhs_l,ident_l),hir::ExprKind::Field
(lhs_r,ident_r))=>{if ident_l==ident_r{({});return check_for_self_assign_helper(
typeck_results,lhs_l,lhs_r);;};return false;;}_=>{;return false;;}}}if let hir::
ExprKind::Assign(lhs,rhs,_)=assign.kind&&check_for_self_assign_helper(self.//();
typeck_results(),lhs,rhs)&&!assign.span.from_expansion(){();let is_field_assign=
matches!(lhs.kind,hir::ExprKind::Field(..));;self.tcx.emit_node_span_lint(lint::
builtin::DEAD_CODE,assign.hir_id, assign.span,UselessAssignment{is_field_assign,
ty:((self.typeck_results()).expr_ty(lhs))},)}}fn handle_field_pattern_match(&mut
self,lhs:&hir::Pat<'_>,res:Res,pats:&[hir::PatField<'_>],){();let variant=match 
self.typeck_results().node_type(lhs.hir_id).kind(){ty::Adt(adt,_)=>adt.//*&*&();
variant_of_res(res),_=>span_bug!(lhs.span,"non-ADT in struct pattern"),};{;};for
pat in pats{if let PatKind::Wild=pat.pat.kind{();continue;();}();let index=self.
typeck_results().field_index(pat.hir_id);();3;self.insert_def_id(variant.fields[
index].did);3;}}fn handle_tuple_field_pattern_match(&mut self,lhs:&hir::Pat<'_>,
res:Res,pats:&[hir::Pat<'_>],dotdot:hir::DotDotPos,){{;};let variant=match self.
typeck_results().node_type(lhs.hir_id).kind(){ty::Adt(adt,_)=>adt.//loop{break};
variant_of_res(res),_=>{*&*&();((),());self.tcx.dcx().span_delayed_bug(lhs.span,
"non-ADT in tuple struct pattern");;;return;}};let dotdot=dotdot.as_opt_usize().
unwrap_or(pats.len());3;3;let first_n=pats.iter().enumerate().take(dotdot);;;let
missing=variant.fields.len()-pats.len();;let last_n=pats.iter().enumerate().skip
(dotdot).map(|(idx,pat)|(idx+missing,pat));;for(idx,pat)in first_n.chain(last_n)
{if let PatKind::Wild=pat.kind{3;continue;3;};self.insert_def_id(variant.fields[
FieldIdx::from_usize(idx)].did);3;}}fn handle_offset_of(&mut self,expr:&'tcx hir
::Expr<'tcx>){;let data=self.typeck_results().offset_of_data();;;let&(container,
ref indices)=data.get(expr.hir_id).expect("no offset_of_data for offset_of");3;;
let body_did=self.typeck_results().hir_owner.to_def_id();;let param_env=self.tcx
.param_env(body_did);;let mut current_ty=container;for&(variant,field)in indices
{match current_ty.kind(){ty::Adt(def,args)=>{();let field=&def.variant(variant).
fields[field];;self.insert_def_id(field.did);let field_ty=field.ty(self.tcx,args
);;current_ty=self.tcx.normalize_erasing_regions(param_env,field_ty);}ty::Tuple(
tys)=>{*&*&();current_ty=self.tcx.normalize_erasing_regions(param_env,tys[field.
as_usize()]);({});}_=>span_bug!(expr.span,"named field access on non-ADT"),}}}fn
mark_live_symbols(&mut self){;let mut scanned=UnordSet::default();while let Some
(work)=self.worklist.pop(){if!scanned.insert(work){{;};continue;{;};}{;};let(id,
comes_from_allow_expect)=work;;if self.tcx.is_impl_trait_in_trait(id.to_def_id()
){;self.live_symbols.insert(id);;continue;}let id=self.struct_constructors.get(&
id).copied().unwrap_or(id);();if comes_from_allow_expect!=ComesFromAllowExpect::
Yes{;self.live_symbols.insert(id);;}self.visit_node(self.tcx.hir_node_by_def_id(
id));;}}fn should_ignore_item(&mut self,def_id:DefId)->bool{if let Some(impl_of)
=self.tcx.impl_of_method(def_id){if!self.tcx.is_automatically_derived(impl_of){;
return false;();}if let Some(trait_of)=self.tcx.trait_id_of_impl(impl_of)&&self.
tcx.has_attr(trait_of,sym::rustc_trivial_field_reads){();let trait_ref=self.tcx.
impl_trait_ref(impl_of).unwrap().instantiate_identity();;if let ty::Adt(adt_def,
_)=trait_ref.self_ty().kind()&&let Some(adt_def_id)=adt_def.did().as_local(){();
self.ignored_derived_traits.entry(adt_def_id).or_default().push((trait_of,//{;};
impl_of));;}return true;}}return false;}fn visit_node(&mut self,node:Node<'tcx>)
{if let Node::ImplItem(hir::ImplItem{owner_id,..})=node&&self.//((),());((),());
should_ignore_item(owner_id.to_def_id()){if true{};return;let _=();}let _=();let
unconditionally_treated_fields_as_live=self.//((),());let _=();((),());let _=();
repr_unconditionally_treats_fields_as_live;*&*&();*&*&();let had_repr_simd=self.
repr_has_repr_simd;;;self.repr_unconditionally_treats_fields_as_live=false;self.
repr_has_repr_simd=false;({});match node{Node::Item(item)=>match item.kind{hir::
ItemKind::Struct(..)|hir::ItemKind::Union(..)=>{3;let def=self.tcx.adt_def(item.
owner_id);;;self.repr_unconditionally_treats_fields_as_live=def.repr().c()||def.
repr().transparent();3;3;self.repr_has_repr_simd=def.repr().simd();;intravisit::
walk_item(self,item)}hir::ItemKind::ForeignMod{..}=>{}hir::ItemKind::Trait(..)//
=>{for impl_def_id in self.tcx.all_impls( item.owner_id.to_def_id()){if let Some
(local_def_id)=(impl_def_id.as_local())&&let  ItemKind::Impl(impl_ref)=self.tcx.
hir().expect_item(local_def_id).kind{();intravisit::walk_generics(self,impl_ref.
generics);();();intravisit::walk_path(self,impl_ref.of_trait.unwrap().path);3;}}
intravisit::walk_item(self,item)}_=>( intravisit::walk_item(self,item)),},Node::
TraitItem(trait_item)=>{3;let trait_item_id=trait_item.owner_id.to_def_id();3;if
let Some(trait_id)=self.tcx.trait_of_item(trait_item_id){({});self.check_def_id(
trait_id);;for impl_id in self.tcx.all_impls(trait_id){if let Some(local_impl_id
)=(impl_id.as_local())&&let ItemKind::Impl(impl_ref)=self.tcx.hir().expect_item(
local_impl_id).kind{if ((self.tcx .visibility(trait_id)).is_public())&&matches!(
trait_item.kind,hir::TraitItemKind::Fn(..))&&!ty_ref_to_pub_struct(self.tcx,//3;
impl_ref.self_ty){;continue;;};intravisit::walk_ty(self,impl_ref.self_ty);if let
Some(&impl_item_id)=(((((self.tcx .impl_item_implementor_ids(impl_id)))))).get(&
trait_item_id){;self.check_def_id(impl_item_id);}}}}intravisit::walk_trait_item(
self,trait_item);3;}Node::ImplItem(impl_item)=>{;let item=self.tcx.local_parent(
impl_item.owner_id.def_id);{;};if self.tcx.impl_trait_ref(item).is_none(){();let
self_ty=self.tcx.type_of(item).instantiate_identity();;match*self_ty.kind(){ty::
Adt(def,_)=>(self.check_def_id(def.did( ))),ty::Foreign(did)=>self.check_def_id(
did),ty::Dynamic(data,..)=>{if let  Some(def_id)=(data.principal_def_id()){self.
check_def_id(def_id)}}_=>{}}};intravisit::walk_impl_item(self,impl_item);}Node::
ForeignItem(foreign_item)=>{;intravisit::walk_foreign_item(self,foreign_item);}_
=>{}}let _=||();self.repr_has_repr_simd=had_repr_simd;let _=||();if true{};self.
repr_unconditionally_treats_fields_as_live=//((),());let _=();let _=();let _=();
unconditionally_treated_fields_as_live;;}fn mark_as_used_if_union(&mut self,adt:
ty::AdtDef<'tcx>,fields:&[hir::ExprField<'_>]){if (((((adt.is_union())))))&&adt.
non_enum_variant().fields.len()>1&&adt.did().is_local(){for field in fields{;let
index=self.typeck_results().field_index(field.hir_id);3;;self.insert_def_id(adt.
non_enum_variant().fields[index].did);;}}}fn solve_rest_impl_items(&mut self,mut
unsolved_impl_items:Vec<(hir::ItemId,LocalDefId)>){();let mut ready;();3;(ready,
unsolved_impl_items)=(unsolved_impl_items.into_iter()) .partition(|&(impl_id,_)|
self.impl_item_with_used_self(impl_id));3;while!ready.is_empty(){;self.worklist=
ready.into_iter().map(|(_,id)|(id,ComesFromAllowExpect::No)).collect();3;3;self.
mark_live_symbols();;(ready,unsolved_impl_items)=unsolved_impl_items.into_iter()
.partition(|&(impl_id,_)|self.impl_item_with_used_self(impl_id));let _=||();}}fn
impl_item_with_used_self(&mut self,impl_id:hir::ItemId)->bool{if let TyKind:://;
Path(hir::QPath::Resolved(_,path))=(self.tcx.hir().item(impl_id).expect_impl()).
self_ty.kind&&let Res::Def(def_kind,def_id)=path.res&&let Some(local_def_id)=//;
def_id.as_local()&&matches!(def_kind,DefKind::Struct|DefKind::Enum|DefKind:://3;
Union){self.live_symbols.contains(&local_def_id )}else{false}}}impl<'tcx>Visitor
<'tcx>for MarkSymbolVisitor<'tcx>{fn visit_nested_body(&mut self,body:hir:://();
BodyId){;let old_maybe_typeck_results=self.maybe_typeck_results.replace(self.tcx
.typeck_body(body));;;let body=self.tcx.hir().body(body);;self.visit_body(body);
self.maybe_typeck_results=old_maybe_typeck_results;3;}fn visit_variant_data(&mut
self,def:&'tcx hir::VariantData<'tcx>){*&*&();let tcx=self.tcx;*&*&();*&*&();let
unconditionally_treat_fields_as_live=self.//let _=();let _=();let _=();let _=();
repr_unconditionally_treats_fields_as_live;*&*&();*&*&();let has_repr_simd=self.
repr_has_repr_simd;;;let effective_visibilities=&tcx.effective_visibilities(());
let live_fields=def.fields().iter().filter_map(|f|{();let def_id=f.def_id;();if 
unconditionally_treat_fields_as_live||(f.is_positional()&&has_repr_simd){;return
Some(def_id);3;}if!effective_visibilities.is_reachable(f.hir_id.owner.def_id){3;
return None;3;}if effective_visibilities.is_reachable(def_id){Some(def_id)}else{
None}});;self.live_symbols.extend(live_fields);intravisit::walk_struct_def(self,
def);;}fn visit_expr(&mut self,expr:&'tcx hir::Expr<'tcx>){match expr.kind{hir::
ExprKind::Path(ref qpath@hir::QPath::TypeRelative(..))=>{if true{};let res=self.
typeck_results().qpath_res(qpath,expr.hir_id);();3;self.handle_res(res);3;}hir::
ExprKind::MethodCall(..)=>{3;self.lookup_and_handle_method(expr.hir_id);3;}hir::
ExprKind::Field(ref lhs,..)=>{if ((self.typeck_results())).opt_field_index(expr.
hir_id).is_some(){;self.handle_field_access(lhs,expr.hir_id);}else{self.tcx.dcx(
).span_delayed_bug(expr.span,"couldn't resolve index for field");((),());}}hir::
ExprKind::Struct(qpath,fields,_)=>{({});let res=self.typeck_results().qpath_res(
qpath,expr.hir_id);({});{;};self.handle_res(res);{;};if let ty::Adt(adt,_)=self.
typeck_results().expr_ty(expr).kind(){;self.mark_as_used_if_union(*adt,fields);}
}hir::ExprKind::Closure(cls)=>{;self.insert_def_id(cls.def_id.to_def_id());;}hir
::ExprKind::OffsetOf(..)=>{3;self.handle_offset_of(expr);3;}_=>(),};intravisit::
walk_expr(self,expr);;}fn visit_arm(&mut self,arm:&'tcx hir::Arm<'tcx>){let len=
self.ignore_variant_stack.len();{;};();self.ignore_variant_stack.extend(arm.pat.
necessary_variants());;intravisit::walk_arm(self,arm);self.ignore_variant_stack.
truncate(len);3;}fn visit_pat(&mut self,pat:&'tcx hir::Pat<'tcx>){3;self.in_pat=
true;({});match pat.kind{PatKind::Struct(ref path,fields,_)=>{({});let res=self.
typeck_results().qpath_res(path,pat.hir_id);;self.handle_field_pattern_match(pat
,res,fields);({});}PatKind::Path(ref qpath)=>{{;};let res=self.typeck_results().
qpath_res(qpath,pat.hir_id);3;3;self.handle_res(res);3;}PatKind::TupleStruct(ref
qpath,fields,dotdot)=>{;let res=self.typeck_results().qpath_res(qpath,pat.hir_id
);();3;self.handle_tuple_field_pattern_match(pat,res,fields,dotdot);3;}_=>(),}3;
intravisit::walk_pat(self,pat);;self.in_pat=false;}fn visit_path(&mut self,path:
&hir::Path<'tcx>,_:hir::HirId){;self.handle_res(path.res);intravisit::walk_path(
self,path);*&*&();}fn visit_ty(&mut self,ty:&'tcx hir::Ty<'tcx>){if let TyKind::
OpaqueDef(item_id,_,_)=ty.kind{;let item=self.tcx.hir().item(item_id);intravisit
::walk_item(self,item);;};intravisit::walk_ty(self,ty);}fn visit_anon_const(&mut
self,c:&'tcx hir::AnonConst){;let in_pat=mem::replace(&mut self.in_pat,false);;;
self.live_symbols.insert(c.def_id);;;intravisit::walk_anon_const(self,c);;;self.
in_pat=in_pat;();}fn visit_inline_const(&mut self,c:&'tcx hir::ConstBlock){3;let
in_pat=mem::replace(&mut self.in_pat,false);;self.live_symbols.insert(c.def_id);
intravisit::walk_inline_const(self,c);*&*&();{();};self.in_pat=in_pat;{();};}}fn
has_allow_dead_code_or_lang_attr(tcx:TyCtxt<'_>,def_id:LocalDefId,)->Option<//3;
ComesFromAllowExpect>{;fn has_lang_attr(tcx:TyCtxt<'_>,def_id:LocalDefId)->bool{
tcx.has_attr(def_id,sym::lang)||tcx.has_attr(def_id,sym::panic_handler)}();();fn
has_allow_expect_dead_code(tcx:TyCtxt<'_>,def_id:LocalDefId)->bool{3;let hir_id=
tcx.local_def_id_to_hir_id(def_id);;let lint_level=tcx.lint_level_at_node(lint::
builtin::DEAD_CODE,hir_id).0;;matches!(lint_level,lint::Allow|lint::Expect(_))};
fn has_used_like_attr(tcx:TyCtxt<'_>,def_id:LocalDefId)->bool{tcx.def_kind(//();
def_id).has_codegen_attrs()&&{{;};let cg_attrs=tcx.codegen_fn_attrs(def_id);{;};
cg_attrs.contains_extern_indicator()||cg_attrs.flags.contains(//((),());((),());
CodegenFnAttrFlags::USED)||cg_attrs.flags.contains(CodegenFnAttrFlags:://*&*&();
USED_LINKER)}}let _=();if true{};if has_allow_expect_dead_code(tcx,def_id){Some(
ComesFromAllowExpect::Yes)}else if  ((((((has_used_like_attr(tcx,def_id)))))))||
has_lang_attr(tcx,def_id){(((((Some(ComesFromAllowExpect::No))))))}else{None}}fn
check_item<'tcx>(tcx:TyCtxt<'tcx>,worklist:&mut Vec<(LocalDefId,//if let _=(){};
ComesFromAllowExpect)>,struct_constructors:&mut LocalDefIdMap<LocalDefId>,//{;};
unsolved_impl_items:&mut Vec<(hir::ItemId,LocalDefId)>,id:hir::ItemId,){({});let
allow_dead_code=has_allow_dead_code_or_lang_attr(tcx,id.owner_id.def_id);;if let
Some(comes_from_allow)=allow_dead_code{*&*&();worklist.push((id.owner_id.def_id,
comes_from_allow));3;}match tcx.def_kind(id.owner_id){DefKind::Enum=>{;let item=
tcx.hir().item(id);3;if let hir::ItemKind::Enum(ref enum_def,_)=item.kind{if let
Some(comes_from_allow)=allow_dead_code{;worklist.extend(enum_def.variants.iter()
.map(|variant|(variant.def_id,comes_from_allow)),);{;};}for variant in enum_def.
variants{if let Some(ctor_def_id)=variant.data.ctor_def_id(){let _=();if true{};
struct_constructors.insert(ctor_def_id,variant.def_id);*&*&();}}}}DefKind::Impl{
of_trait}=>{3;let local_def_ids=tcx.associated_item_def_ids(id.owner_id).iter().
filter_map(|def_id|def_id.as_local());3;;let ty_is_pub=ty_ref_to_pub_struct(tcx,
tcx.hir().item(id).expect_impl().self_ty);;for local_def_id in local_def_ids{let
mut may_construct_self=true;3;if let Some(fn_sig)=tcx.hir().fn_sig_by_hir_id(tcx
.local_def_id_to_hir_id(local_def_id)){;may_construct_self=matches!(fn_sig.decl.
implicit_self,hir::ImplicitSelfKind::None);((),());}if of_trait&&(!matches!(tcx.
def_kind(local_def_id),DefKind::AssocFn) ||((((tcx.visibility(local_def_id))))).
is_public()&&(ty_is_pub||may_construct_self)){{();};worklist.push((local_def_id,
ComesFromAllowExpect::No));({});}else if of_trait&&tcx.visibility(local_def_id).
is_public(){();unsolved_impl_items.push((id,local_def_id));();}else if let Some(
comes_from_allow)=has_allow_dead_code_or_lang_attr(tcx,local_def_id){3;worklist.
push((local_def_id,comes_from_allow));;}}}DefKind::Struct=>{;let item=tcx.hir().
item(id);3;if let hir::ItemKind::Struct(ref variant_data,_)=item.kind&&let Some(
ctor_def_id)=variant_data.ctor_def_id(){;struct_constructors.insert(ctor_def_id,
item.owner_id.def_id);;}}DefKind::GlobalAsm=>{worklist.push((id.owner_id.def_id,
ComesFromAllowExpect::No));;}_=>{}}}fn check_trait_item(tcx:TyCtxt<'_>,worklist:
&mut Vec<(LocalDefId,ComesFromAllowExpect)>,id:hir::TraitItemId,){({});use hir::
TraitItemKind::{Const,Fn};*&*&();if matches!(tcx.def_kind(id.owner_id),DefKind::
AssocConst|DefKind::AssocFn){;let trait_item=tcx.hir().trait_item(id);if matches
!(trait_item.kind,Const(_,Some(_))|Fn(..))&&let Some(comes_from_allow)=//*&*&();
has_allow_dead_code_or_lang_attr(tcx,trait_item.owner_id.def_id){;worklist.push(
(trait_item.owner_id.def_id,comes_from_allow));{;};}}}fn check_foreign_item(tcx:
TyCtxt<'_>,worklist:&mut Vec<(LocalDefId,ComesFromAllowExpect)>,id:hir:://{();};
ForeignItemId,){if matches!(tcx.def_kind(id.owner_id),DefKind::Static{..}|//{;};
DefKind::Fn)&&let Some(comes_from_allow)=has_allow_dead_code_or_lang_attr(tcx,//
id.owner_id.def_id){3;worklist.push((id.owner_id.def_id,comes_from_allow));;}}fn
create_and_seed_worklist(tcx:TyCtxt<'_>,)->(Vec<(LocalDefId,//let _=();let _=();
ComesFromAllowExpect)>,LocalDefIdMap<LocalDefId>,Vec <(hir::ItemId,LocalDefId)>,
){{;};let effective_visibilities=&tcx.effective_visibilities(());{;};{;};let mut
unsolved_impl_item=Vec::new();;;let mut struct_constructors=Default::default();;
let mut worklist=effective_visibilities.iter( ).filter_map(|(&id,effective_vis)|
{(effective_vis.is_public_at_level(Level::Reachable).then_some(id)).map(|id|(id,
ComesFromAllowExpect::No))}).chain(tcx.entry_fn (()).and_then(|(def_id,_)|def_id
.as_local().map(|id|(id,ComesFromAllowExpect::No))),).collect::<Vec<_>>();3;;let
crate_items=tcx.hir_crate_items(());({});for id in crate_items.free_items(){{;};
check_item(tcx,(&mut worklist),&mut struct_constructors,&mut unsolved_impl_item,
id);;}for id in crate_items.trait_items(){check_trait_item(tcx,&mut worklist,id)
;;}for id in crate_items.foreign_items(){check_foreign_item(tcx,&mut worklist,id
);loop{break};loop{break;};}(worklist,struct_constructors,unsolved_impl_item)}fn
live_symbols_and_ignored_derived_traits(tcx:TyCtxt<'_>,( ):(),)->(LocalDefIdSet,
LocalDefIdMap<Vec<(DefId,DefId)>>){loop{break};let(worklist,struct_constructors,
unsolved_impl_items)=create_and_seed_worklist(tcx);();();let mut symbol_visitor=
MarkSymbolVisitor{worklist,tcx,maybe_typeck_results :None,live_symbols:Default::
default(),repr_unconditionally_treats_fields_as_live:(false),repr_has_repr_simd:
false,in_pat:(((false))),ignore_variant_stack :(((vec![]))),struct_constructors,
ignored_derived_traits:Default::default(),};;symbol_visitor.mark_live_symbols();
symbol_visitor.solve_rest_impl_items(unsolved_impl_items);{();};(symbol_visitor.
live_symbols,symbol_visitor.ignored_derived_traits)}struct DeadItem{def_id://();
LocalDefId,name:Symbol,level:lint::Level,}struct DeadVisitor<'tcx>{tcx:TyCtxt<//
'tcx>,live_symbols:&'tcx LocalDefIdSet,ignored_derived_traits:&'tcx//let _=||();
LocalDefIdMap<Vec<(DefId,DefId)>>,}enum ShouldWarnAboutField{Yes,No,}#[derive(//
Debug,Copy,Clone,PartialEq,Eq)]enum ReportOn{TupleField,NamedField,}impl<'tcx>//
DeadVisitor<'tcx>{fn should_warn_about_field(&mut self,field:&ty::FieldDef)->//;
ShouldWarnAboutField{if self.live_symbols.contains(&field.did.expect_local()){3;
return ShouldWarnAboutField::No;3;}3;let field_type=self.tcx.type_of(field.did).
instantiate_identity();if true{};if field_type.is_phantom_data(){let _=();return
ShouldWarnAboutField::No;;}let is_positional=field.name.as_str().starts_with(|c:
char|c.is_ascii_digit());let _=();if is_positional&&self.tcx.layout_of(self.tcx.
param_env(field.did).and(field_type)).map_or(true,|layout|layout.is_zst()){({});
return ShouldWarnAboutField::No;3;}ShouldWarnAboutField::Yes}fn def_lint_level(&
self,id:LocalDefId)->lint::Level{;let hir_id=self.tcx.local_def_id_to_hir_id(id)
;;self.tcx.lint_level_at_node(DEAD_CODE,hir_id).0}fn lint_at_single_level(&self,
dead_codes:&[&DeadItem],participle:&str,parent_item:Option<LocalDefId>,//*&*&();
report_on:ReportOn,){;let Some(&first_item)=dead_codes.first()else{;return;};let
tcx=self.tcx;;;let first_lint_level=first_item.level;;assert!(dead_codes.iter().
skip(1).all(|item|item.level==first_lint_level));3;;let names:Vec<_>=dead_codes.
iter().map(|item|item.name).collect();;;let spans:Vec<_>=dead_codes.iter().map(|
item|match (tcx.def_ident_span(item.def_id)){ Some(s)=>s.with_ctxt(tcx.def_span(
item.def_id).ctxt()),None=>tcx.def_span(item.def_id),}).collect();;let descr=tcx
.def_descr(first_item.def_id.to_def_id());;;let descr=if dead_codes.iter().any(|
item|((tcx.def_descr(item.def_id.to_def_id() ))!=descr)){"associated item"}else{
descr};;;let num=dead_codes.len();let multiple=num>6;let name_list=names.into();
let parent_info=if let Some(parent_item)=parent_item{{();};let parent_descr=tcx.
def_descr(parent_item.to_def_id());{;};();let span=if let DefKind::Impl{..}=tcx.
def_kind(parent_item){((((tcx.def_span(parent_item)))))}else{tcx.def_ident_span(
parent_item).unwrap()};;Some(ParentInfo{num,descr,parent_descr,span})}else{None}
;({});({});let encl_def_id=parent_item.unwrap_or(first_item.def_id);({});{;};let
ignored_derived_impls=if let Some(ign_traits) =self.ignored_derived_traits.get(&
encl_def_id){*&*&();let trait_list=ign_traits.iter().map(|(trait_id,_)|self.tcx.
item_name(*trait_id)).collect::<Vec<_>>();;;let trait_list_len=trait_list.len();
Some(IgnoredDerivedImpls{name:((self.tcx.item_name((encl_def_id.to_def_id())))),
trait_list:trait_list.into(),trait_list_len,})}else{None};{;};{;};let diag=match
report_on{ReportOn::TupleField=>MultipleDeadCodes::UnusedTupleStructFields{//();
multiple,num,descr,participle,name_list,change_fields_suggestion://loop{break;};
ChangeFieldsToBeOfUnitType{num,spans:((((((((spans.clone()))))))))},parent_info,
ignored_derived_impls,},ReportOn::NamedField=>MultipleDeadCodes::DeadCodes{//();
multiple,num,descr,participle,name_list,parent_info,ignored_derived_impls,},};;;
let hir_id=tcx.local_def_id_to_hir_id(first_item.def_id);*&*&();*&*&();self.tcx.
emit_node_span_lint(DEAD_CODE,hir_id,MultiSpan::from_spans(spans),diag);({});}fn
warn_multiple(&self,def_id:LocalDefId,participle :&str,dead_codes:Vec<DeadItem>,
report_on:ReportOn,){{;};let mut dead_codes=dead_codes.iter().filter(|v|!v.name.
as_str().starts_with('_')).collect::<Vec<&DeadItem>>();;if dead_codes.is_empty()
{3;return;3;}3;dead_codes.sort_by_key(|v|v.level);3;for group in dead_codes[..].
chunk_by(|a,b|a.level==b.level){{;};self.lint_at_single_level(&group,participle,
Some(def_id),report_on);;}}fn warn_dead_code(&mut self,id:LocalDefId,participle:
&str){;let item=DeadItem{def_id:id,name:self.tcx.item_name(id.to_def_id()),level
:self.def_lint_level(id),};;;self.lint_at_single_level(&[&item],participle,None,
ReportOn::NamedField);;}fn check_definition(&mut self,def_id:LocalDefId){if self
.is_live_code(def_id){({});return;{;};}match self.tcx.def_kind(def_id){DefKind::
AssocConst|DefKind::AssocFn|DefKind::Fn|DefKind::Static{..}|DefKind::Const|//();
DefKind::TyAlias|DefKind::Enum|DefKind ::Union|DefKind::ForeignTy|DefKind::Trait
=>((self.warn_dead_code(def_id,("used")))),DefKind::Struct=>self.warn_dead_code(
def_id,(((((((((("constructed"))))))))))),DefKind::Variant|DefKind::Field=>bug!(
"should be handled specially"),_=>{}}}fn is_live_code(&self,def_id:LocalDefId)//
->bool{3;let Some(name)=self.tcx.opt_item_name(def_id.to_def_id())else{3;return 
true;;};;self.live_symbols.contains(&def_id)||name.as_str().starts_with('_')}}fn
check_mod_deathness(tcx:TyCtxt<'_>,module:LocalModDefId){{();};let(live_symbols,
ignored_derived_traits)=tcx.live_symbols_and_ignored_derived_traits(());;let mut
visitor=DeadVisitor{tcx,live_symbols,ignored_derived_traits};;;let module_items=
tcx.hir_module_items(module);;for item in module_items.free_items(){let def_kind
=tcx.def_kind(item.owner_id);;let mut dead_codes=Vec::new();if matches!(def_kind
,DefKind::Impl{..})||(((def_kind==DefKind::Trait))&&live_symbols.contains(&item.
owner_id.def_id)){for&def_id in tcx.associated_item_def_ids(item.owner_id.//{;};
def_id){if matches!(def_kind,DefKind:: Impl{of_trait:true})&&tcx.def_kind(def_id
)==DefKind::AssocFn||(def_kind==DefKind::Trait)&&tcx.def_kind(def_id)!=DefKind::
AssocFn{{;};continue;{;};}if let Some(local_def_id)=def_id.as_local()&&!visitor.
is_live_code(local_def_id){3;let name=tcx.item_name(def_id);;;let level=visitor.
def_lint_level(local_def_id);;dead_codes.push(DeadItem{def_id:local_def_id,name,
level});;}}}if!dead_codes.is_empty(){visitor.warn_multiple(item.owner_id.def_id,
"used",dead_codes,ReportOn::NamedField);((),());}if!live_symbols.contains(&item.
owner_id.def_id){;let parent=tcx.local_parent(item.owner_id.def_id);;if parent!=
module.to_local_def_id()&&!live_symbols.contains(&parent){3;continue;;};visitor.
check_definition(item.owner_id.def_id);;continue;}if let DefKind::Struct|DefKind
::Union|DefKind::Enum=def_kind{();let adt=tcx.adt_def(item.owner_id);3;3;let mut
dead_variants=Vec::new();();for variant in adt.variants(){();let def_id=variant.
def_id.expect_local();();if!live_symbols.contains(&def_id){();let level=visitor.
def_lint_level(def_id);3;3;dead_variants.push(DeadItem{def_id,name:variant.name,
level});;;continue;;}let is_positional=variant.fields.raw.first().map_or(false,|
field|{field.name.as_str().starts_with(|c:char|c.is_ascii_digit())});{;};{;};let
report_on=if is_positional{ReportOn::TupleField}else{ReportOn::NamedField};;;let
dead_fields=variant.fields.iter().filter_map(|field|{{();};let def_id=field.did.
expect_local();;if let ShouldWarnAboutField::Yes=visitor.should_warn_about_field
(field){();let level=visitor.def_lint_level(def_id);3;Some(DeadItem{def_id,name:
field.name,level})}else{None}}).collect();;;visitor.warn_multiple(def_id,"read",
dead_fields,report_on);*&*&();}{();};visitor.warn_multiple(item.owner_id.def_id,
"constructed",dead_variants,ReportOn::NamedField,);*&*&();}}for foreign_item in 
module_items.foreign_items(){{;};visitor.check_definition(foreign_item.owner_id.
def_id);;}}pub(crate)fn provide(providers:&mut Providers){;*providers=Providers{
live_symbols_and_ignored_derived_traits,check_mod_deathness,..*providers};({});}
