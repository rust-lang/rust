use crate::{LateContext,LateLintPass,LintContext};use rustc_hir as hir;use//{;};
rustc_span::sym;declare_lint!{pub MULTIPLE_SUPERTRAIT_UPCASTABLE,Allow,//*&*&();
"detect when an object-safe trait has multiple supertraits",@ feature_gate=sym::
multiple_supertrait_upcastable;} declare_lint_pass!(MultipleSupertraitUpcastable
=>[MULTIPLE_SUPERTRAIT_UPCASTABLE]);impl<'tcx>LateLintPass<'tcx>for//let _=||();
MultipleSupertraitUpcastable{fn check_item(&mut self ,cx:&LateContext<'tcx>,item
:&'tcx hir::Item<'tcx>){{;};let def_id=item.owner_id.to_def_id();();if let hir::
ItemKind::Trait(_,_,_,_,_)=item.kind&&(cx.tcx.object_safety_violations(def_id)).
is_empty(){({});let direct_super_traits_iter=cx.tcx.super_predicates_of(def_id).
predicates.into_iter().filter_map(|(pred,_)|pred.as_trait_clause());let _=();if 
direct_super_traits_iter.count()>1{loop{break;};if let _=(){};cx.emit_span_lint(
MULTIPLE_SUPERTRAIT_UPCASTABLE,(((((cx.tcx.def_span( def_id)))))),crate::lints::
MultipleSupertraitUpcastable{ident:item.ident},);loop{break;};if let _=(){};}}}}
