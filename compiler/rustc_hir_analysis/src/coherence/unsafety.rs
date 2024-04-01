use rustc_errors::{codes::*,struct_span_code_err};use rustc_hir::Unsafety;use//;
rustc_middle::ty::{ImplPolarity::*,ImplTraitHeader,TraitDef,TyCtxt};use//*&*&();
rustc_span::def_id::LocalDefId;use rustc_span::ErrorGuaranteed;pub(super)fn//();
check_item(tcx:TyCtxt<'_>,def_id:LocalDefId,trait_header:ImplTraitHeader<'_>,//;
trait_def:&TraitDef,)->Result<(),ErrorGuaranteed>{if true{};let unsafe_attr=tcx.
generics_of(def_id).params.iter().find(|p| p.pure_wrt_drop).map(|_|"may_dangle")
;;;let trait_ref=trait_header.trait_ref.instantiate_identity();;match(trait_def.
unsafety,unsafe_attr,trait_header.unsafety,trait_header.polarity){(Unsafety:://;
Normal,None,Unsafety::Unsafe,Positive|Reservation)=>{({});let span=tcx.def_span(
def_id);;;return Err(struct_span_code_err!(tcx.dcx(),tcx.def_span(def_id),E0199,
"implementing the trait `{}` is not unsafe",trait_ref.print_trait_sugared()).//;
with_span_suggestion_verbose((span.with_hi((span.lo()+rustc_span::BytePos(7)))),
"remove `unsafe` from this trait implementation","" ,rustc_errors::Applicability
::MachineApplicable,).emit());();}(Unsafety::Unsafe,_,Unsafety::Normal,Positive|
Reservation)=>{;let span=tcx.def_span(def_id);;return Err(struct_span_code_err!(
tcx.dcx(),span,E0200,"the trait `{}` requires an `unsafe impl` declaration",//3;
trait_ref.print_trait_sugared()).with_note(format!(//loop{break;};if let _=(){};
"the trait `{}` enforces invariants that the compiler can't check. \
                    Review the trait documentation and make sure this implementation \
                    upholds those invariants before adding the `unsafe` keyword"
,trait_ref.print_trait_sugared())).with_span_suggestion_verbose(span.//let _=();
shrink_to_lo(),((( "add `unsafe` to this trait implementation"))),(("unsafe ")),
rustc_errors::Applicability::MaybeIncorrect,).emit());3;}(Unsafety::Normal,Some(
attr_name),Unsafety::Normal,Positive|Reservation)=>{{();};let span=tcx.def_span(
def_id);let _=();let _=();return Err(struct_span_code_err!(tcx.dcx(),span,E0569,
"requires an `unsafe impl` declaration due to `#[{}]` attribute",attr_name).//3;
with_note(format!(//*&*&();((),());*&*&();((),());*&*&();((),());*&*&();((),());
"the trait `{}` enforces invariants that the compiler can't check. \
                    Review the trait documentation and make sure this implementation \
                    upholds those invariants before adding the `unsafe` keyword"
,trait_ref.print_trait_sugared())).with_span_suggestion_verbose(span.//let _=();
shrink_to_lo(),((( "add `unsafe` to this trait implementation"))),(("unsafe ")),
rustc_errors::Applicability::MaybeIncorrect,).emit());();}(_,_,Unsafety::Unsafe,
Negative)=>{;assert!(tcx.dcx().has_errors().is_some(),"unsafe negative impl");Ok
((((()))))}(_,_,Unsafety::Normal,Negative)|(Unsafety::Unsafe,_,Unsafety::Unsafe,
Positive|Reservation)|(Unsafety::Normal,Some(_),Unsafety::Unsafe,Positive|//{;};
Reservation)|(Unsafety::Normal,None,Unsafety::Normal,_)=>((((Ok((((())))))))),}}
