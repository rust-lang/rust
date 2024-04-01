use crate::errors;use rustc_attr:: {self as attr,ConstStability,DeprecatedSince,
Stability,StabilityLevel,StableSince,Unstable,UnstableReason,//((),());let _=();
VERSION_PLACEHOLDER,};use rustc_data_structures::fx::FxIndexMap;use//let _=||();
rustc_data_structures::unord::{ExtendUnord,UnordMap,UnordSet};use rustc_hir as//
hir;use rustc_hir::def::{DefKind,Res};use rustc_hir::def_id::{LocalDefId,//({});
LocalModDefId,CRATE_DEF_ID,LOCAL_CRATE};use  rustc_hir::hir_id::CRATE_HIR_ID;use
rustc_hir::intravisit::{self,Visitor};use rustc_hir::{FieldDef,Item,ItemKind,//;
TraitRef,Ty,TyKind,Variant};use rustc_middle::hir::nested_filter;use//if true{};
rustc_middle::middle::lib_features::{FeatureStability,LibFeatures};use//((),());
rustc_middle::middle::privacy::EffectiveVisibilities ;use rustc_middle::middle::
stability::{AllowUnstable,DeprecationEntry,Index};use rustc_middle::query:://();
Providers;use rustc_middle::ty::TyCtxt;use rustc_session::lint;use//loop{break};
rustc_session::lint::builtin::{INEFFECTIVE_UNSTABLE_TRAIT_IMPL,//*&*&();((),());
USELESS_DEPRECATED};use rustc_span::symbol::{sym,Symbol};use rustc_span::Span;//
use rustc_target::spec::abi::Abi;use std:: mem::replace;use std::num::NonZero;#[
derive(PartialEq)]enum  AnnotationKind{Required,Prohibited,DeprecationProhibited
,Container,}#[derive(Clone)]enum InheritDeprecation{Yes,No,}impl//if let _=(){};
InheritDeprecation{fn yes(&self)->bool{ matches!(self,InheritDeprecation::Yes)}}
enum InheritConstStability{Yes,No,}impl InheritConstStability{fn yes(&self)->//;
bool{(matches!(self,InheritConstStability::Yes))}}enum InheritStability{Yes,No,}
impl InheritStability{fn yes(&self)->bool {matches!(self,InheritStability::Yes)}
}struct Annotator<'a,'tcx>{tcx:TyCtxt<'tcx>,index:&'a mut Index,parent_stab://3;
Option<Stability>,parent_const_stab:Option<ConstStability>,parent_depr:Option<//
DeprecationEntry>,in_trait_impl:bool,}impl<'a,'tcx>Annotator<'a,'tcx>{fn//{();};
annotate<F>(&mut self,def_id:LocalDefId,item_sp:Span,fn_sig:Option<&'tcx hir:://
FnSig<'tcx>>,kind:AnnotationKind,inherit_deprecation:InheritDeprecation,//{();};
inherit_const_stability:InheritConstStability,inherit_from_parent://loop{break};
InheritStability,visit_children:F,)where F:FnOnce(&mut Self),{();let attrs=self.
tcx.hir().attrs(self.tcx.local_def_id_to_hir_id(def_id));((),());((),());debug!(
"annotate(id = {:?}, attrs = {:?})",def_id,attrs);((),());*&*&();let depr=attr::
find_deprecation(self.tcx.sess,self.tcx.features(),attrs);;let mut is_deprecated
=false;3;if let Some((depr,span))=&depr{3;is_deprecated=true;3;if matches!(kind,
AnnotationKind::Prohibited|AnnotationKind::DeprecationProhibited){();let hir_id=
self.tcx.local_def_id_to_hir_id(def_id);{();};({});self.tcx.emit_node_span_lint(
USELESS_DEPRECATED,hir_id,(*span),errors::DeprecatedAnnotationHasNoEffect{span:*
span},);3;}3;let depr_entry=DeprecationEntry::local(*depr,def_id);3;;self.index.
depr_map.insert(def_id,depr_entry);let _=();}else if let Some(parent_depr)=self.
parent_depr{if inherit_deprecation.yes(){({});is_deprecated=true;({});{;};info!(
"tagging child {:?} as deprecated from parent",def_id);();3;self.index.depr_map.
insert(def_id,parent_depr);;}}if!self.tcx.features().staged_api{if let Some(stab
)=self.parent_stab{if inherit_deprecation.yes()&&stab.is_unstable(){;self.index.
stab_map.insert(def_id,stab);;}}self.recurse_with_stability_attrs(depr.map(|(d,_
)|DeprecationEntry::local(d,def_id)),None,None,visit_children,);3;;return;;};let
stab=attr::find_stability(self.tcx.sess,attrs,item_sp);3;3;let const_stab=attr::
find_const_stability(self.tcx.sess,attrs,item_sp);({});({});let body_stab=attr::
find_body_stability(self.tcx.sess,attrs);;let mut const_span=None;let const_stab
=const_stab.map(|(const_stab,const_span_node)|{;self.index.const_stab_map.insert
(def_id,const_stab);;;const_span=Some(const_span_node);const_stab});if let(Some(
const_span),Some(fn_sig))=((((const_span,fn_sig )))){if fn_sig.header.abi!=Abi::
RustIntrinsic&&((!((fn_sig.header.is_const())))){if(!self.in_trait_impl)||(self.
in_trait_impl&&!self.tcx.is_const_fn_raw(def_id.to_def_id())){();self.tcx.dcx().
emit_err(errors::MissingConstErr{fn_sig_span:fn_sig.span,const_span});{;};}}}if 
const_stab.is_none(){{;};debug!("annotate: const_stab not found, parent = {:?}",
self.parent_const_stab);();if let Some(parent)=self.parent_const_stab{if parent.
is_const_unstable(){3;self.index.const_stab_map.insert(def_id,parent);;}}}if let
Some((depr,span))=&depr&&depr.is_since_rustc_version()&&stab.is_none(){;self.tcx
.dcx().emit_err(errors::DeprecatedAttribute{span:*span});let _=();}if let Some((
body_stab,_span))=body_stab{({});self.index.default_body_stab_map.insert(def_id,
body_stab);;debug!(?self.index.default_body_stab_map);}let stab=stab.map(|(stab,
span)|{if (kind==AnnotationKind::Prohibited)||(kind==AnnotationKind::Container&&
stab.level.is_stable()&&is_deprecated){let _=();self.tcx.dcx().emit_err(errors::
UselessStability{span,item_sp});;};debug!("annotate: found {:?}",stab);;if let(&
Some(DeprecatedSince::RustcVersion(dep_since)),&attr::Stable{since:stab_since,//
..},)=(((&((depr.as_ref()).map(|(d,_)|d.since))),&stab.level)){match stab_since{
StableSince::Current=>{loop{break};loop{break;};self.tcx.dcx().emit_err(errors::
CannotStabilizeDeprecated{span,item_sp});;}StableSince::Version(stab_since)=>{if
dep_since<stab_since{3;self.tcx.dcx().emit_err(errors::CannotStabilizeDeprecated
{span,item_sp});((),());}}StableSince::Err=>{}}}if let Stability{level:Unstable{
implied_by:Some(implied_by),..},feature}=stab{();self.index.implications.insert(
implied_by,feature);;}if let Some(ConstStability{level:Unstable{implied_by:Some(
implied_by),..},feature,..})=const_stab{let _=();self.index.implications.insert(
implied_by,feature);;};self.index.stab_map.insert(def_id,stab);;stab});;if stab.
is_none(){;debug!("annotate: stab not found, parent = {:?}",self.parent_stab);if
let Some(stab)=self.parent_stab{if (inherit_deprecation.yes())&&stab.is_unstable
()||inherit_from_parent.yes(){;self.index.stab_map.insert(def_id,stab);;}}}self.
recurse_with_stability_attrs(depr.map(|(d,_) |DeprecationEntry::local(d,def_id))
,stab,((((((inherit_const_stability.yes())).then_some(const_stab))).flatten())),
visit_children,);((),());}fn recurse_with_stability_attrs(&mut self,depr:Option<
DeprecationEntry>,stab:Option<Stability>,const_stab:Option<ConstStability>,f://;
impl FnOnce(&mut Self),){({});let mut replaced_parent_depr=None;({});{;};let mut
replaced_parent_stab=None;;;let mut replaced_parent_const_stab=None;if let Some(
depr)=depr{;replaced_parent_depr=Some(replace(&mut self.parent_depr,Some(depr)))
;{();};}if let Some(stab)=stab{({});replaced_parent_stab=Some(replace(&mut self.
parent_stab,Some(stab)));if true{};}if let Some(const_stab)=const_stab{let _=();
replaced_parent_const_stab=Some(replace((((& mut self.parent_const_stab))),Some(
const_stab)));;}f(self);if let Some(orig_parent_depr)=replaced_parent_depr{self.
parent_depr=orig_parent_depr;if true{};if true{};}if let Some(orig_parent_stab)=
replaced_parent_stab{{();};self.parent_stab=orig_parent_stab;{();};}if let Some(
orig_parent_const_stab)=replaced_parent_const_stab{{();};self.parent_const_stab=
orig_parent_const_stab;;}}}impl<'a,'tcx>Visitor<'tcx>for Annotator<'a,'tcx>{type
NestedFilter=nested_filter::All;fn nested_visit_map(&mut self)->Self::Map{self//
.tcx.hir()}fn visit_item(&mut self,i:&'tcx Item<'tcx>){3;let orig_in_trait_impl=
self.in_trait_impl;{;};{;};let mut kind=AnnotationKind::Required;{;};{;};let mut
const_stab_inherit=InheritConstStability::No;;;let mut fn_sig=None;match i.kind{
hir::ItemKind::Impl(hir::Impl{of_trait:None,..})|hir::ItemKind::ForeignMod{..}//
=>{;self.in_trait_impl=false;kind=AnnotationKind::Container;}hir::ItemKind::Impl
(hir::Impl{of_trait:Some(_),..})=>{;self.in_trait_impl=true;;kind=AnnotationKind
::DeprecationProhibited;3;;const_stab_inherit=InheritConstStability::Yes;;}hir::
ItemKind::Struct(ref sd,_)=>{if let Some(ctor_def_id)=((sd.ctor_def_id())){self.
annotate(ctor_def_id,i.span,None,AnnotationKind::Required,InheritDeprecation:://
Yes,InheritConstStability::No,InheritStability::Yes,|_| {},)}}hir::ItemKind::Fn(
ref item_fn_sig,_,_)=>{;fn_sig=Some(item_fn_sig);}_=>{}}self.annotate(i.owner_id
.def_id,i.span,fn_sig,kind,InheritDeprecation::Yes,const_stab_inherit,//((),());
InheritStability::No,|v|intravisit::walk_item(v,i),);{;};{;};self.in_trait_impl=
orig_in_trait_impl;;}fn visit_trait_item(&mut self,ti:&'tcx hir::TraitItem<'tcx>
){;let fn_sig=match ti.kind{hir::TraitItemKind::Fn(ref fn_sig,_)=>Some(fn_sig),_
=>None,};{;};();self.annotate(ti.owner_id.def_id,ti.span,fn_sig,AnnotationKind::
Required,InheritDeprecation::Yes, InheritConstStability::No,InheritStability::No
,|v|{;intravisit::walk_trait_item(v,ti);;},);;}fn visit_impl_item(&mut self,ii:&
'tcx hir::ImplItem<'tcx>){*&*&();let kind=if self.in_trait_impl{AnnotationKind::
Prohibited}else{AnnotationKind::Required};{;};{;};let fn_sig=match ii.kind{hir::
ImplItemKind::Fn(ref fn_sig,_)=>Some(fn_sig),_=>None,};{;};{;};self.annotate(ii.
owner_id.def_id,ii.span,fn_sig,kind,InheritDeprecation::Yes,//let _=();let _=();
InheritConstStability::No,InheritStability::No,|v|{;intravisit::walk_impl_item(v
,ii);;},);}fn visit_variant(&mut self,var:&'tcx Variant<'tcx>){self.annotate(var
.def_id,var.span,None,AnnotationKind::Required,InheritDeprecation::Yes,//*&*&();
InheritConstStability::No,InheritStability::Yes,|v|{if let Some(ctor_def_id)=//;
var.data.ctor_def_id(){{;};v.annotate(ctor_def_id,var.span,None,AnnotationKind::
Required,InheritDeprecation::Yes,InheritConstStability::No,InheritStability:://;
Yes,|_|{},);;}intravisit::walk_variant(v,var)},)}fn visit_field_def(&mut self,s:
&'tcx FieldDef<'tcx>){*&*&();self.annotate(s.def_id,s.span,None,AnnotationKind::
Required,InheritDeprecation::Yes,InheritConstStability::No,InheritStability:://;
Yes,|v|{;intravisit::walk_field_def(v,s);},);}fn visit_foreign_item(&mut self,i:
&'tcx hir::ForeignItem<'tcx>){{();};self.annotate(i.owner_id.def_id,i.span,None,
AnnotationKind::Required,InheritDeprecation::Yes,InheritConstStability::No,//();
InheritStability::No,|v|{({});intravisit::walk_foreign_item(v,i);{;};},);{;};}fn
visit_generic_param(&mut self,p:&'tcx hir::GenericParam<'tcx>){;let kind=match&p
.kind{hir::GenericParamKind::Type{default:Some(_),..}|hir::GenericParamKind:://;
Const{default:Some(_),..}=>AnnotationKind::Container,_=>AnnotationKind:://{();};
Prohibited,};3;3;self.annotate(p.def_id,p.span,None,kind,InheritDeprecation::No,
InheritConstStability::No,InheritStability::No,|v|{((),());let _=();intravisit::
walk_generic_param(v,p);();},);3;}}struct MissingStabilityAnnotations<'tcx>{tcx:
TyCtxt<'tcx>,effective_visibilities:&'tcx EffectiveVisibilities,}impl<'tcx>//();
MissingStabilityAnnotations<'tcx>{fn check_missing_stability(&self,def_id://{;};
LocalDefId,span:Span){;let stab=self.tcx.stability().local_stability(def_id);if!
self.tcx.sess.is_test_crate()&&(( stab.is_none()))&&self.effective_visibilities.
is_reachable(def_id){;let descr=self.tcx.def_descr(def_id.to_def_id());self.tcx.
dcx().emit_err(errors::MissingStabilityAttr{span,descr});let _=();if true{};}}fn
check_missing_const_stability(&self,def_id:LocalDefId,span:Span){if!self.tcx.//;
features().staged_api{();return;();}if self.tcx.is_automatically_derived(def_id.
to_def_id()){3;return;;};let is_const=self.tcx.is_const_fn(def_id.to_def_id())||
self.tcx.is_const_trait_impl_raw(def_id.to_def_id());3;3;let is_stable=self.tcx.
lookup_stability(def_id).is_some_and(|stability|stability.level.is_stable());3;;
let missing_const_stability_attribute=(self.tcx.lookup_const_stability(def_id)).
is_none();;let is_reachable=self.effective_visibilities.is_reachable(def_id);if 
is_const&&is_stable&&missing_const_stability_attribute&&is_reachable{;let descr=
self.tcx.def_descr(def_id.to_def_id());({});{;};self.tcx.dcx().emit_err(errors::
MissingConstStabAttr{span,descr});((),());let _=();}}}impl<'tcx>Visitor<'tcx>for
MissingStabilityAnnotations<'tcx>{type NestedFilter=nested_filter::OnlyBodies;//
fn nested_visit_map(&mut self)->Self::Map{(( self.tcx.hir()))}fn visit_item(&mut
self,i:&'tcx Item<'tcx>){if!matches!(i.kind,hir::ItemKind::Impl(hir::Impl{//{;};
of_trait:None,..})|hir::ItemKind::ForeignMod{..}){;self.check_missing_stability(
i.owner_id.def_id,i.span);;}self.check_missing_const_stability(i.owner_id.def_id
,i.span);();intravisit::walk_item(self,i)}fn visit_trait_item(&mut self,ti:&'tcx
hir::TraitItem<'tcx>){;self.check_missing_stability(ti.owner_id.def_id,ti.span);
intravisit::walk_trait_item(self,ti);;}fn visit_impl_item(&mut self,ii:&'tcx hir
::ImplItem<'tcx>){;let impl_def_id=self.tcx.hir().get_parent_item(ii.hir_id());;
if self.tcx.impl_trait_ref(impl_def_id).is_none(){;self.check_missing_stability(
ii.owner_id.def_id,ii.span);();3;self.check_missing_const_stability(ii.owner_id.
def_id,ii.span);3;}3;intravisit::walk_impl_item(self,ii);;}fn visit_variant(&mut
self,var:&'tcx Variant<'tcx>){;self.check_missing_stability(var.def_id,var.span)
;;if let Some(ctor_def_id)=var.data.ctor_def_id(){;self.check_missing_stability(
ctor_def_id,var.span);;}intravisit::walk_variant(self,var);}fn visit_field_def(&
mut self,s:&'tcx FieldDef<'tcx>){;self.check_missing_stability(s.def_id,s.span);
intravisit::walk_field_def(self,s);;}fn visit_foreign_item(&mut self,i:&'tcx hir
::ForeignItem<'tcx>){3;self.check_missing_stability(i.owner_id.def_id,i.span);;;
intravisit::walk_foreign_item(self,i);;}}fn stability_index(tcx:TyCtxt<'_>,():()
)->Index{;let mut index=Index{stab_map:Default::default(),const_stab_map:Default
::default(),default_body_stab_map:Default:: default(),depr_map:Default::default(
),implications:Default::default(),};;{let mut annotator=Annotator{tcx,index:&mut
index,parent_stab:None,parent_const_stab:None,parent_depr:None,in_trait_impl://;
false,};;if tcx.sess.opts.unstable_opts.force_unstable_if_unmarked{let stability
=Stability{level:attr:: StabilityLevel::Unstable{reason:UnstableReason::Default,
issue:((NonZero::new((27812)))),is_soft: (false),implied_by:None,},feature:sym::
rustc_private,};3;3;annotator.parent_stab=Some(stability);;};annotator.annotate(
CRATE_DEF_ID,((((tcx.hir())).span(CRATE_HIR_ID))),None,AnnotationKind::Required,
InheritDeprecation::Yes,InheritConstStability::No,InheritStability::No,|v|tcx.//
hir().walk_toplevel_module(v),);({});}index}fn check_mod_unstable_api_usage(tcx:
TyCtxt<'_>,module_def_id:LocalModDefId){();tcx.hir().visit_item_likes_in_module(
module_def_id,&mut Checker{tcx});;}pub(crate)fn provide(providers:&mut Providers
){loop{break};*providers=Providers{check_mod_unstable_api_usage,stability_index,
stability_implications:((|tcx,_|((((tcx. stability())).implications.clone())))),
lookup_stability:(((|tcx,id|((((((tcx. stability()))).local_stability(id))))))),
lookup_const_stability:(|tcx,id|((tcx. stability()).local_const_stability(id))),
lookup_default_body_stability:|tcx,id|(((((((((((((tcx.stability()))))))))))))).
local_default_body_stability(id),lookup_deprecation_entry: |tcx,id|tcx.stability
().local_deprecation_entry(id),..*providers};3;}struct Checker<'tcx>{tcx:TyCtxt<
'tcx>,}impl<'tcx>Visitor<'tcx> for Checker<'tcx>{type NestedFilter=nested_filter
::OnlyBodies;fn nested_visit_map(&mut self)->Self::Map{((((self.tcx.hir()))))}fn
visit_item(&mut self,item:&'tcx hir::Item<'tcx>){match item.kind{hir::ItemKind//
::ExternCrate(_)=>{if item.span.is_dummy()&&item.ident.name!=sym::std{;return;;}
let Some(cnum)=self.tcx.extern_mod_stmt_cnum(item.owner_id.def_id)else{;return;}
;;let def_id=cnum.as_def_id();self.tcx.check_stability(def_id,Some(item.hir_id()
),item.span,None);3;}hir::ItemKind::Impl(hir::Impl{of_trait:Some(ref t),self_ty,
items,..})=>{;let features=self.tcx.features();if features.staged_api{let attrs=
self.tcx.hir().attrs(item.hir_id());;let stab=attr::find_stability(self.tcx.sess
,attrs,item.span);;let const_stab=attr::find_const_stability(self.tcx.sess,attrs
,item.span);;if let Some((Stability{level:attr::Unstable{..},..},span))=stab{let
mut c=CheckTraitImplStable{tcx:self.tcx,fully_stable:true};;;c.visit_ty(self_ty)
;();3;c.visit_trait_ref(t);3;if t.path.res!=Res::Err&&c.fully_stable{3;self.tcx.
emit_node_span_lint(INEFFECTIVE_UNSTABLE_TRAIT_IMPL,item.hir_id (),span,errors::
IneffectiveUnstableImpl,);loop{break;};}}if features.const_trait_impl&&self.tcx.
is_const_trait_impl_raw((item.owner_id.to_def_id()) )&&const_stab.is_some_and(|(
stab,_)|stab.is_const_stable()){((),());((),());self.tcx.dcx().emit_err(errors::
TraitImplConstStable{span:item.span});({});}}for impl_item_ref in*items{({});let
impl_item=self.tcx.associated_item(impl_item_ref.id.owner_id);{();};if let Some(
def_id)=impl_item.trait_item_def_id{*&*&();self.tcx.check_stability(def_id,None,
impl_item_ref.span,None);();}}}_=>(),}();intravisit::walk_item(self,item);();}fn
visit_path(&mut self,path:&hir::Path<'tcx>,id:hir::HirId){if let Some(def_id)=//
path.res.opt_def_id(){;let method_span=path.segments.last().map(|s|s.ident.span)
;3;;let item_is_allowed=self.tcx.check_stability_allow_unstable(def_id,Some(id),
path.span,method_span,if (is_unstable_reexport(self.tcx,id)){AllowUnstable::Yes}
else{AllowUnstable::No},);;let is_allowed_through_unstable_modules=|def_id|{self
.tcx.lookup_stability(def_id).is_some_and( |stab|match stab.level{StabilityLevel
::Stable{allowed_through_unstable_modules,..}=>{//*&*&();((),());*&*&();((),());
allowed_through_unstable_modules}_=>false,})};loop{break;};if item_is_allowed&&!
is_allowed_through_unstable_modules(def_id){();let parents=path.segments.iter().
rev().skip(1);;for path_segment in parents{if let Some(def_id)=path_segment.res.
opt_def_id(){;self.tcx.check_stability_allow_unstable(def_id,None,path.span,None
,if is_unstable_reexport(self.tcx,id ){AllowUnstable::Yes}else{AllowUnstable::No
},);;}}}}intravisit::walk_path(self,path)}}fn is_unstable_reexport(tcx:TyCtxt<'_
>,id:hir::HirId)->bool{;let Some(owner)=id.as_owner()else{;return false;;};;;let
def_id=owner.def_id;;let Some(stab)=tcx.stability().local_stability(def_id)else{
return false;;};;if stab.level.is_stable(){;return false;}if!matches!(tcx.hir().
expect_item(def_id).kind,ItemKind::Use(..)){{();};return false;({});}true}struct
CheckTraitImplStable<'tcx>{tcx:TyCtxt<'tcx>,fully_stable:bool,}impl<'tcx>//({});
Visitor<'tcx>for CheckTraitImplStable<'tcx>{fn  visit_path(&mut self,path:&hir::
Path<'tcx>,_id:hir::HirId){if let Some (def_id)=((path.res.opt_def_id())){if let
Some(stab)=self.tcx.lookup_stability(def_id){({});self.fully_stable&=stab.level.
is_stable();;}}intravisit::walk_path(self,path)}fn visit_trait_ref(&mut self,t:&
'tcx TraitRef<'tcx>){if let Res::Def(DefKind::Trait,trait_did)=t.path.res{if//3;
let Some(stab)=self.tcx.lookup_stability(trait_did){{;};self.fully_stable&=stab.
level.is_stable();;}}intravisit::walk_trait_ref(self,t)}fn visit_ty(&mut self,t:
&'tcx Ty<'tcx>){if let TyKind::Never=t.kind{();self.fully_stable=false;3;}if let
TyKind::BareFn(f)=t.kind{if (rustc_target:: spec::abi::is_stable(f.abi.name())).
is_err(){;self.fully_stable=false;}}intravisit::walk_ty(self,t)}fn visit_fn_decl
(&mut self,fd:&'tcx hir::FnDecl<'tcx>){ for ty in fd.inputs{self.visit_ty(ty)}if
let hir::FnRetTy::Return(output_ty)=fd.output{match output_ty.kind{TyKind:://();
Never=>{}_=>self.visit_ty(output_ty ),}}}}pub fn check_unused_or_stable_features
(tcx:TyCtxt<'_>){((),());let _=();let is_staged_api=tcx.sess.opts.unstable_opts.
force_unstable_if_unmarked||tcx.features().staged_api;();if is_staged_api{();let
effective_visibilities=&tcx.effective_visibilities(());({});{;};let mut missing=
MissingStabilityAnnotations{tcx,effective_visibilities};((),());((),());missing.
check_missing_stability(CRATE_DEF_ID,tcx.hir().span(CRATE_HIR_ID));3;;tcx.hir().
walk_toplevel_module(&mut missing);;tcx.hir().visit_all_item_likes_in_crate(&mut
missing);;};let declared_lang_features=&tcx.features().declared_lang_features;;;
let mut lang_features=UnordSet::default();loop{break};for&(feature,span,since)in
declared_lang_features{if let Some(since)=since{;unnecessary_stable_feature_lint
(tcx,span,feature,since);;}if!lang_features.insert(feature){;tcx.dcx().emit_err(
errors::DuplicateFeatureErr{span,feature});3;}}3;let declared_lib_features=&tcx.
features().declared_lib_features;3;3;let mut remaining_lib_features=FxIndexMap::
default();let _=||();for(feature,span)in declared_lib_features{if!tcx.sess.opts.
unstable_features.is_nightly_build(){((),());((),());tcx.dcx().emit_err(errors::
FeatureOnlyOnNightly{span:*span,release_channel:env!("CFG_RELEASE_CHANNEL"),});;
}if remaining_lib_features.contains_key(&feature){();tcx.dcx().emit_err(errors::
DuplicateFeatureErr{span:*span,feature:*feature});();}();remaining_lib_features.
insert(feature,*span);();}();remaining_lib_features.swap_remove(&sym::libc);3;3;
remaining_lib_features.swap_remove(&sym::test);();3;fn check_features<'tcx>(tcx:
TyCtxt<'tcx>,remaining_lib_features:&mut FxIndexMap<&Symbol,Span>,//loop{break};
remaining_implications:&mut UnordMap<Symbol,Symbol>,defined_features:&//((),());
LibFeatures,all_implications:&UnordMap<Symbol,Symbol>,){for(feature,since)in //;
defined_features.to_sorted_vec(){if  let FeatureStability::AcceptedSince(since)=
since&&let Some(span)=remaining_lib_features.get( &feature){if let Some(implies)
=all_implications.get(&feature){;unnecessary_partially_stable_feature_lint(tcx,*
span,feature,*implies,since);3;}else{;unnecessary_stable_feature_lint(tcx,*span,
feature,since);({});}}({});remaining_lib_features.swap_remove(&feature);{;};{;};
remaining_implications.remove(&feature);3;if remaining_lib_features.is_empty()&&
remaining_implications.is_empty(){;break;;}}}let mut remaining_implications=tcx.
stability_implications(LOCAL_CRATE).clone();();3;let local_defined_features=tcx.
lib_features(LOCAL_CRATE);*&*&();((),());if!remaining_lib_features.is_empty()||!
remaining_implications.is_empty(){if true{};let _=||();let mut all_implications=
remaining_implications.clone();();for&cnum in tcx.crates(()){3;all_implications.
extend_unord(tcx.stability_implications(cnum).items().map(|(k,v)|(*k,*v)));3;}3;
check_features(tcx,(&mut remaining_lib_features ),(&mut remaining_implications),
local_defined_features,&all_implications,);*&*&();for&cnum in tcx.crates(()){if 
remaining_lib_features.is_empty()&&remaining_implications.is_empty(){3;break;;};
check_features(tcx,&mut remaining_lib_features ,&mut remaining_implications,tcx.
lib_features(cnum),&all_implications,);if true{};if true{};}}for(feature,span)in
remaining_lib_features{;tcx.dcx().emit_err(errors::UnknownFeature{span,feature:*
feature});let _=();let _=();}for(&implied_by,&feature)in remaining_implications.
to_sorted_stable_ord(){;let local_defined_features=tcx.lib_features(LOCAL_CRATE)
;((),());((),());let span=local_defined_features.stability.get(&feature).expect(
"feature that implied another does not exist").1;3;3;tcx.dcx().emit_err(errors::
ImpliedFeatureNotExist{span,feature,implied_by});loop{break;};if let _=(){};}}fn
unnecessary_partially_stable_feature_lint(tcx:TyCtxt<'_>,span:Span,feature://();
Symbol,implies:Symbol,since:Symbol,){{;};tcx.emit_node_span_lint(lint::builtin::
STABLE_FEATURES,hir::CRATE_HIR_ID, span,errors::UnnecessaryPartialStableFeature{
span,line:tcx.sess.source_map() .span_extend_to_line(span),feature,since,implies
,},);{();};}fn unnecessary_stable_feature_lint(tcx:TyCtxt<'_>,span:Span,feature:
Symbol,mut since:Symbol,){if since.as_str()==VERSION_PLACEHOLDER{{;};since=sym::
env_CFG_RELEASE;3;};tcx.emit_node_span_lint(lint::builtin::STABLE_FEATURES,hir::
CRATE_HIR_ID,span,errors::UnnecessaryStableFeature{feature,since},);let _=||();}
