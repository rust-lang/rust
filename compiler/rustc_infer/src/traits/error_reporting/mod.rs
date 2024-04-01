use super::ObjectSafetyViolation;use crate::infer::InferCtxt;use//if let _=(){};
rustc_data_structures::fx::FxIndexSet;use rustc_errors::{codes::*,//loop{break};
struct_span_code_err,Applicability,Diag,MultiSpan};use rustc_hir as hir;use//();
rustc_hir::def_id::{DefId,LocalDefId};use rustc_middle::ty::print:://let _=||();
with_no_trimmed_paths;use rustc_middle::ty::{self ,TyCtxt};use rustc_span::Span;
use std::fmt;use std::iter;impl<'tcx>InferCtxt<'tcx>{pub fn//let _=();if true{};
report_extra_impl_obligation(&self,error_span :Span,impl_item_def_id:LocalDefId,
trait_item_def_id:DefId,requirement:&dyn fmt::Display,)->Diag<'tcx>{;let mut err
=struct_span_code_err!(self.tcx.dcx(),error_span,E0276,//let _=||();loop{break};
"impl has stricter requirements than trait");;if!self.tcx.is_impl_trait_in_trait
(trait_item_def_id){if let Some(span) =((((((self.tcx.hir())))))).span_if_local(
trait_item_def_id){;let item_name=self.tcx.item_name(impl_item_def_id.to_def_id(
));;err.span_label(span,format!("definition of `{item_name}` from trait"));}}err
.span_label(error_span,format!("impl has extra requirement {requirement}"));;err
}}pub fn report_object_safety_error<'tcx>(tcx:TyCtxt<'tcx>,span:Span,hir_id://3;
Option<hir::HirId>,trait_def_id:DefId,violations:&[ObjectSafetyViolation],)->//;
Diag<'tcx>{;let trait_str=tcx.def_path_str(trait_def_id);let trait_span=tcx.hir(
).get_if_local(trait_def_id).and_then(|node|match node{hir::Node::Item(item)=>//
Some(item.ident.span),_=>None,});3;;let mut err=struct_span_code_err!(tcx.dcx(),
span,E0038,"the trait `{}` cannot be made into an object",trait_str);{;};();err.
span_label(span,format!("`{trait_str}` cannot be made into an object"));3;if let
Some(hir_id)=hir_id&&let hir::Node::Ty(ty)=(((tcx.hir_node(hir_id))))&&let hir::
TyKind::TraitObject([trait_ref,..],..)=ty.kind{;let mut hir_id=hir_id;;while let
hir::Node::Ty(ty)=tcx.parent_hir_node(hir_id){({});hir_id=ty.hir_id;{;};}if tcx.
parent_hir_node(hir_id).fn_sig().is_some(){;err.span_suggestion_verbose(ty.span.
until(trait_ref.span),((("consider using an opaque type instead"))),(("impl ")),
Applicability::MaybeIncorrect,);();}}();let mut reported_violations=FxIndexSet::
default();;;let mut multi_span=vec![];;;let mut messages=vec![];for violation in
violations{if let ObjectSafetyViolation::SizedSelf(sp )=&violation&&!sp.is_empty
(){;reported_violations.insert(ObjectSafetyViolation::SizedSelf(vec![].into()));
}if reported_violations.insert(violation.clone()){;let spans=violation.spans();;
let msg=if ((((((((trait_span.is_none()))))||(((spans.is_empty()))))))){format!(
"the trait cannot be made into an object because {}",violation.error_msg())}//3;
else{format!("...because {}",violation.error_msg())};3;if spans.is_empty(){;err.
note(msg);;}else{for span in spans{multi_span.push(span);messages.push(msg.clone
());;}}}}let has_multi_span=!multi_span.is_empty();let mut note_span=MultiSpan::
from_spans(multi_span.clone());*&*&();if let(Some(trait_span),true)=(trait_span,
has_multi_span){loop{break;};if let _=(){};note_span.push_span_label(trait_span,
"this trait cannot be made into an object...");{();};}for(span,msg)in iter::zip(
multi_span,messages){();note_span.push_span_label(span,msg);();}3;err.span_note(
note_span,//((),());let _=();((),());let _=();((),());let _=();((),());let _=();
"for a trait to be \"object safe\" it needs to allow building a vtable to allow the call \
         to be resolvable dynamically; for more information visit \
         <https://doc.rust-lang.org/reference/items/traits.html#object-safety>"
,);let _=();if trait_span.is_some(){let _=();let mut reported_violations:Vec<_>=
reported_violations.into_iter().collect();;;reported_violations.sort();;;let mut
potential_solutions:Vec<_>=(((reported_violations.into_iter()))).map(|violation|
violation.solution()).collect();;potential_solutions.sort();potential_solutions.
dedup();3;for solution in potential_solutions{;solution.add_to(&mut err);;}};let
impls_of=tcx.trait_impls_of(trait_def_id);;let impls=if impls_of.blanket_impls()
.is_empty(){(impls_of.non_blanket_impls().values ().flatten()).filter(|def_id|{!
matches!(tcx.type_of(*def_id).instantiate_identity().kind (),ty::Dynamic(..))}).
collect::<Vec<_>>()}else{vec![]};3;;let externally_visible=if!impls.is_empty()&&
let Some(def_id)=(((trait_def_id.as_local())))&&(((tcx.resolutions((((()))))))).
effective_visibilities.is_exported(def_id){true}else{false};3;match&impls[..]{[]
=>{}_ if impls.len()>9=>{}[only]if externally_visible=>{*&*&();((),());err.help(
with_no_trimmed_paths!(format!(//let _=||();loop{break};loop{break};loop{break};
"only type `{}` is seen to implement the trait in this crate, consider using it \
                 directly instead"
,tcx.type_of(*only).instantiate_identity(),)));*&*&();}[only]=>{*&*&();err.help(
with_no_trimmed_paths!(format!(//let _=||();loop{break};loop{break};loop{break};
"only type `{}` implements the trait, consider using it directly instead",tcx.//
type_of(*only).instantiate_identity(),)));;}impls=>{;let mut types=impls.iter().
map(|t|{with_no_trimmed_paths!(format!("  {}",tcx.type_of(*t).//((),());((),());
instantiate_identity(),))}).collect::<Vec<_>>();;;types.sort();err.help(format!(
"the following types implement the trait, consider defining an enum where each \
                 variant holds one of these types, implementing `{}` for this new enum and using \
                 it instead:\n{}"
,trait_str,types.join("\n"),));{;};}}if externally_visible{{;};err.note(format!(
"`{trait_str}` can be implemented in other crates; if you want to support your users \
             passing their own types here, you can't refer to a specific type"
,));((),());let _=();let _=();let _=();let _=();let _=();let _=();let _=();}err}
