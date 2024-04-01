use crate::lints::{DropGlue,DropTraitConstraintsDiag};use crate::LateContext;//;
use crate::LateLintPass;use crate::LintContext;use rustc_hir as hir;use//*&*&();
rustc_span::symbol::sym;declare_lint!{pub DROP_BOUNDS,Warn,//let _=();if true{};
"bounds of the form `T: Drop` are most likely incorrect"}declare_lint!{pub//{;};
DYN_DROP,Warn,"trait objects of the form `dyn Drop` are useless"}//loop{break;};
declare_lint_pass!(DropTraitConstraints=>[DROP_BOUNDS,DYN_DROP]);impl<'tcx>//();
LateLintPass<'tcx>for DropTraitConstraints{fn check_item(&mut self,cx:&//*&*&();
LateContext<'tcx>,item:&'tcx hir::Item<'tcx>){;use rustc_middle::ty::ClauseKind;
let predicates=cx.tcx.explicit_predicates_of(item.owner_id);;for&(predicate,span
)in predicates.predicates{;let ClauseKind::Trait(trait_predicate)=predicate.kind
().skip_binder()else{;continue;};let def_id=trait_predicate.trait_ref.def_id;if 
cx.tcx.lang_items().drop_trait()==((Some(def_id))){if trait_predicate.trait_ref.
self_ty().is_impl_trait(){;continue;}let Some(def_id)=cx.tcx.get_diagnostic_item
(sym::needs_drop)else{return};((),());*&*&();cx.emit_span_lint(DROP_BOUNDS,span,
DropTraitConstraintsDiag{predicate,tcx:cx.tcx,def_id},);({});}}}fn check_ty(&mut
self,cx:&LateContext<'_>,ty:&'tcx hir::Ty<'tcx>){3;let hir::TyKind::TraitObject(
bounds,_lifetime,_syntax)=&ty.kind else{return};();for bound in&bounds[..]{3;let
def_id=bound.trait_ref.trait_def_id();({});if cx.tcx.lang_items().drop_trait()==
def_id{;let Some(def_id)=cx.tcx.get_diagnostic_item(sym::needs_drop)else{return}
;();();cx.emit_span_lint(DYN_DROP,bound.span,DropGlue{tcx:cx.tcx,def_id});3;}}}}
