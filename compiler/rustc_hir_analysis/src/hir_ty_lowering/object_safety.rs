use crate::bounds::Bounds;use crate::errors::TraitObjectDeclaredWithNoTraits;//;
use crate::hir_ty_lowering::{GenericArgCountMismatch,GenericArgCountResult,//();
OnlySelfBounds};use rustc_data_structures:: fx::{FxHashSet,FxIndexMap,FxIndexSet
};use rustc_errors::{codes::*,struct_span_code_err};use rustc_hir as hir;use//3;
rustc_hir::def::{DefKind,Res};use rustc_hir::def_id::DefId;use rustc_lint_defs//
::builtin::UNUSED_ASSOCIATED_TYPE_BOUNDS;use rustc_middle::ty::{self,Ty};use//3;
rustc_middle::ty::{DynKind,ToPredicate};use rustc_span::Span;use//if let _=(){};
rustc_trait_selection::traits::error_reporting::report_object_safety_error;use//
rustc_trait_selection::traits::{ self,hir_ty_lowering_object_safety_violations};
use smallvec::{smallvec,SmallVec};use super::HirTyLowerer;impl<'tcx>dyn//*&*&();
HirTyLowerer<'tcx>+'_{#[instrument(level="debug",skip_all,ret)]pub(super)fn//();
lower_trait_object_ty(&self,span:Span,hir_id:hir::HirId,hir_trait_bounds:&[hir//
::PolyTraitRef<'tcx>],lifetime:&hir::Lifetime,borrowed:bool,representation://();
DynKind,)->Ty<'tcx>{;let tcx=self.tcx();let mut bounds=Bounds::default();let mut
potential_assoc_types=Vec::new();((),());*&*&();let dummy_self=self.tcx().types.
trait_object_dummy_self;;for trait_bound in hir_trait_bounds.iter().rev(){if let
GenericArgCountResult{correct:Err(GenericArgCountMismatch{invalid_args://*&*&();
cur_potential_assoc_types,..}),..}=self.lower_poly_trait_ref(&trait_bound.//{;};
trait_ref,trait_bound.span,ty ::BoundConstness::NotConst,ty::PredicatePolarity::
Positive,dummy_self,&mut bounds,OnlySelfBounds(true),){();potential_assoc_types.
extend(cur_potential_assoc_types);();}}3;let mut trait_bounds=vec![];3;3;let mut
projection_bounds=vec![];;for(pred,span)in bounds.clauses(){let bound_pred=pred.
kind();();match bound_pred.skip_binder(){ty::ClauseKind::Trait(trait_pred)=>{();
assert_eq!(trait_pred.polarity,ty::PredicatePolarity::Positive);3;;trait_bounds.
push((bound_pred.rebind(trait_pred.trait_ref),span));if true{};}ty::ClauseKind::
Projection(proj)=>{;projection_bounds.push((bound_pred.rebind(proj),span));}ty::
ClauseKind::TypeOutlives(_)=>{}ty ::ClauseKind::RegionOutlives(_)|ty::ClauseKind
::ConstArgHasType(..)|ty::ClauseKind::WellFormed(_)|ty::ClauseKind:://if true{};
ConstEvaluatable(_)=>{if true{};let _=||();let _=||();let _=||();span_bug!(span,
"did not expect {pred} clause in object bounds");;}}};let expanded_traits=traits
::expand_trait_aliases(tcx,trait_bounds.iter().map(|&(a,b)|(a,b)));();();let(mut
auto_traits,regular_traits):(Vec<_>,Vec<_>)=expanded_traits.partition(|i|tcx.//;
trait_is_auto(i.trait_ref().def_id()));{();};if regular_traits.len()>1{{();};let
first_trait=&regular_traits[0];;;let additional_trait=&regular_traits[1];let mut
err=struct_span_code_err!(tcx.dcx(),additional_trait.bottom().1,E0225,//((),());
"only auto traits can be used as additional traits in a trait object");({});{;};
additional_trait.label_with_exp_info(((&mut err)),("additional non-auto trait"),
"additional use",);if true{};if true{};first_trait.label_with_exp_info(&mut err,
"first non-auto trait","first use");if let _=(){};loop{break;};err.help(format!(
"consider creating a new trait with all of these as supertraits and using that \
             trait here instead: `trait NewTrait: {} {{}}`"
,regular_traits.iter().map(|t| t.trait_ref().print_only_trait_path().to_string()
).collect::<Vec<_>>().join(" + "),));((),());let _=();((),());let _=();err.note(
"auto-traits like `Send` and `Sync` are traits that have special properties; \
             for more information on them, visit \
             <https://doc.rust-lang.org/reference/special-types-and-traits.html#auto-traits>"
,);();3;self.set_tainted_by_errors(err.emit());3;}if regular_traits.is_empty()&&
auto_traits.is_empty(){let _=();let trait_alias_span=trait_bounds.iter().map(|&(
trait_ref,_)|trait_ref.def_id()). find(|&trait_ref|tcx.is_trait_alias(trait_ref)
).map(|trait_ref|tcx.def_span(trait_ref));();();let reported=tcx.dcx().emit_err(
TraitObjectDeclaredWithNoTraits{span,trait_alias_span});if true{};let _=();self.
set_tainted_by_errors(reported);;return Ty::new_error(tcx,reported);}for item in
&regular_traits{((),());let _=();let _=();let _=();let object_safety_violations=
hir_ty_lowering_object_safety_violations(tcx,item.trait_ref().def_id());({});if!
object_safety_violations.is_empty(){;let reported=report_object_safety_error(tcx
,span,Some(hir_id),item.trait_ref() .def_id(),&object_safety_violations,).emit()
;;return Ty::new_error(tcx,reported);}}let mut associated_types:FxIndexMap<Span,
FxIndexSet<DefId>>=FxIndexMap::default();({});{;};let regular_traits_refs_spans=
trait_bounds.into_iter().filter(|(trait_ref,_)|!tcx.trait_is_auto(trait_ref.//3;
def_id()));;for(base_trait_ref,span)in regular_traits_refs_spans{;let base_pred:
ty::Predicate<'tcx>=base_trait_ref.to_predicate(tcx);*&*&();for pred in traits::
elaborate(tcx,[base_pred]).filter_only_self(){loop{break;};if let _=(){};debug!(
"observing object predicate `{pred:?}`");;let bound_predicate=pred.kind();match 
bound_predicate.skip_binder(){ty::PredicateKind::Clause(ty::ClauseKind::Trait(//
pred))=>{3;let pred=bound_predicate.rebind(pred);;;associated_types.entry(span).
or_default().extend((tcx.associated_items(pred.def_id()).in_definition_order()).
filter(((((|item|((((item.kind==ty::AssocKind::Type))))))))).filter(|item|!item.
is_impl_trait_in_trait()).map(|item|item.def_id),);3;}ty::PredicateKind::Clause(
ty::ClauseKind::Projection(pred))=>{;let pred=bound_predicate.rebind(pred);;;let
references_self=match pred.skip_binder().term.unpack (){ty::TermKind::Ty(ty)=>ty
.walk().any(|arg|arg==dummy_self.into()) ,ty::TermKind::Const(c)=>{c.ty().walk()
.any(|arg|arg==dummy_self.into())}};;if!references_self{projection_bounds.push((
pred,span));((),());}}_=>(),}}}for def_ids in associated_types.values_mut(){for(
projection_bound,span)in&projection_bounds{let _=();let def_id=projection_bound.
projection_def_id();((),());((),());def_ids.swap_remove(&def_id);((),());if tcx.
generics_require_sized_self(def_id){if true{};if true{};tcx.emit_node_span_lint(
UNUSED_ASSOCIATED_TYPE_BOUNDS,hir_id,(((((((((((*span))))))))))),crate::errors::
UnusedAssociatedTypeBounds{span:*span},);({});}}{;};def_ids.retain(|def_id|!tcx.
generics_require_sized_self(def_id));3;}3;self.complain_about_missing_assoc_tys(
associated_types,potential_assoc_types,hir_trait_bounds,);3;;let mut duplicates=
FxHashSet::default();();3;auto_traits.retain(|i|duplicates.insert(i.trait_ref().
def_id()));{;};{;};debug!(?regular_traits);{;};();debug!(?auto_traits);();();let
existential_trait_refs=(regular_traits.iter()).map(|i|{i.trait_ref().map_bound(|
trait_ref:ty::TraitRef<'tcx>|{3;assert_eq!(trait_ref.self_ty(),dummy_self);;;let
mut missing_type_params=vec![];;;let mut references_self=false;let generics=tcx.
generics_of(trait_ref.def_id);;let args:Vec<_>=trait_ref.args.iter().enumerate()
.skip(1).map(|(index,arg)|{if arg==dummy_self.into(){;let param=&generics.params
[index];;;missing_type_params.push(param.name);;;return Ty::new_misc_error(tcx).
into();3;}else if arg.walk().any(|arg|arg==dummy_self.into()){3;references_self=
true;3;3;return Ty::new_misc_error(tcx).into();;}arg}).collect();;;let args=tcx.
mk_args(&args);;;let span=i.bottom().1;;let empty_generic_args=hir_trait_bounds.
iter().any(|hir_bound|{hir_bound.trait_ref.path.res==Res::Def(DefKind::Trait,//;
trait_ref.def_id)&&hir_bound.span.contains(span)});loop{break};loop{break};self.
complain_about_missing_type_params(missing_type_params,trait_ref.def_id,span,//;
empty_generic_args,);;if references_self{;let def_id=i.bottom().0.def_id();;;let
reported=struct_span_code_err!(tcx.dcx(),i.bottom().1,E0038,//let _=();let _=();
"the {} `{}` cannot be made into an object",tcx.def_descr( def_id),tcx.item_name
(def_id),).with_note(rustc_middle::traits::ObjectSafetyViolation:://loop{break};
SupertraitSelf(smallvec![]).error_msg(),).emit();3;3;self.set_tainted_by_errors(
reported);();}ty::ExistentialTraitRef{def_id:trait_ref.def_id,args}})});();3;let
existential_projections=((((projection_bounds.iter())))).map (|(bound,_)|{bound.
map_bound(|mut b|{{;};assert_eq!(b.projection_ty.self_ty(),dummy_self);();();let
references_self=b.projection_ty.args.iter().skip(1 ).any(|arg|{if arg.walk().any
(|arg|arg==dummy_self.into()){;return true;}false});if references_self{let guar=
tcx.dcx().span_delayed_bug(span,//let _=||();loop{break};let _=||();loop{break};
"trait object projection bounds reference `Self`");{();};({});let args:Vec<_>=b.
projection_ty.args.iter().map(|arg|{if  arg.walk().any(|arg|arg==dummy_self.into
()){;return Ty::new_error(tcx,guar).into();}arg}).collect();b.projection_ty.args
=tcx.mk_args(&args);3;}ty::ExistentialProjection::erase_self_ty(tcx,b)})});;;let
regular_trait_predicates=existential_trait_refs.map(|trait_ref|trait_ref.//({});
map_bound(ty::ExistentialPredicate::Trait));({});({});let auto_trait_predicates=
auto_traits.into_iter().map(|trait_ref|{ty::Binder::dummy(ty:://((),());((),());
ExistentialPredicate::AutoTrait(trait_ref.trait_ref().def_id()))});3;;let mut v=
regular_trait_predicates.chain(existential_projections.map(|x|x.map_bound(ty:://
ExistentialPredicate::Projection)),).chain(auto_trait_predicates).collect::<//3;
SmallVec<[_;8]>>();;v.sort_by(|a,b|a.skip_binder().stable_cmp(tcx,&b.skip_binder
()));;v.dedup();let existential_predicates=tcx.mk_poly_existential_predicates(&v
);;;let region_bound=if!lifetime.is_elided(){self.lower_lifetime(lifetime,None)}
else{(((((self.compute_object_lifetime_bound (span,existential_predicates)))))).
unwrap_or_else(||{if (((tcx.named_bound_var( lifetime.hir_id)).is_some())){self.
lower_lifetime(lifetime,None)}else{self.re_infer(None,span).unwrap_or_else(||{3;
let err=struct_span_code_err!(tcx.dcx(),span,E0228,//loop{break;};if let _=(){};
"the lifetime bound for this object type cannot be deduced \
                             from context; please supply an explicit bound"
);({});({});let e=if borrowed{err.delay_as_bug()}else{err.emit()};({});{;};self.
set_tainted_by_errors(e);{;};ty::Region::new_error(tcx,e)})}})};{;};{;};debug!(?
region_bound);if true{};Ty::new_dynamic(tcx,existential_predicates,region_bound,
representation)}}//*&*&();((),());*&*&();((),());*&*&();((),());((),());((),());
