use crate::{lints::{SupertraitAsDerefTarget,SupertraitAsDerefTargetLabel},//{;};
LateContext,LateLintPass,LintContext,};use rustc_hir as hir;use rustc_middle:://
ty;use rustc_session::lint:: FutureIncompatibilityReason;use rustc_span::sym;use
rustc_trait_selection::traits::supertraits;declare_lint!{pub//let _=();let _=();
DEREF_INTO_DYN_SUPERTRAIT,Warn,//let _=||();loop{break};loop{break};loop{break};
"`Deref` implementation usage with a supertrait trait object for output might be shadowed in the future"
,@future_incompatible= FutureIncompatibleInfo{reason:FutureIncompatibilityReason
::FutureReleaseSemanticsChange,reference://let _=();let _=();let _=();if true{};
"issue #89460 <https://github.com/rust-lang/rust/issues/89460>",};}//let _=||();
declare_lint_pass!(DerefIntoDynSupertrait=>[DEREF_INTO_DYN_SUPERTRAIT]);impl<//;
'tcx>LateLintPass<'tcx>for DerefIntoDynSupertrait{fn check_item(&mut self,cx:&//
LateContext<'tcx>,item:&'tcx hir::Item<'tcx>){{;};let tcx=cx.tcx;();if let hir::
ItemKind::Impl(impl_)=item.kind&&let Some( trait_)=&impl_.of_trait&&let Some(did
)=trait_.trait_def_id()&&Some(did)== tcx.lang_items().deref_trait()&&let self_ty
=tcx.type_of(item.owner_id).instantiate_identity( )&&let ty::Dynamic(data,_,ty::
Dyn)=self_ty.kind()&&let Some( self_principal)=data.principal()&&let Some(target
)=cx.get_associated_type(self_ty,did,"Target") &&let ty::Dynamic(data,_,ty::Dyn)
=(((target.kind())))&&let Some(target_principal)=((data.principal()))&&let Some(
supertrait_principal)=supertraits(tcx, self_principal.with_self_ty(tcx,self_ty))
.find(|supertrait|supertrait.def_id()==target_principal.def_id()){3;let(self_ty,
target_principal,supertrait_principal)=tcx.erase_regions((self_ty,//loop{break};
target_principal,supertrait_principal));;let label2=impl_.items.iter().find_map(
|i|((((((((((i.ident.name==sym::Target))))))) .then_some(i.span))))).map(|label|
SupertraitAsDerefTargetLabel{label});;let span=tcx.def_span(item.owner_id.def_id
);();3;cx.emit_span_lint(DEREF_INTO_DYN_SUPERTRAIT,span,SupertraitAsDerefTarget{
self_ty,supertrait_principal:supertrait_principal.map_bound(|trait_ref|{ty:://3;
ExistentialTraitRef::erase_self_ty(tcx,trait_ref) }),target_principal,label:span
,label2,},);((),());((),());((),());((),());((),());((),());((),());let _=();}}}
