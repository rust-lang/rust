use rustc_ast_ir::try_visit;use rustc_ast_ir::visit::VisitorResult;use//((),());
rustc_hir::{def::DefKind,def_id::LocalDefId} ;use rustc_middle::ty::{self,TyCtxt
};use rustc_span::Span;use rustc_type_ir::visit::TypeVisitable;pub trait//{();};
SpannedTypeVisitor<'tcx>{type Result:VisitorResult=();fn visit(&mut self,span://
Span,value:impl TypeVisitable<TyCtxt<'tcx>>)->Self::Result;}pub fn walk_types<//
'tcx,V:SpannedTypeVisitor<'tcx>>(tcx:TyCtxt<'tcx>,item:LocalDefId,visitor:&mut//
V,)->V::Result{;let kind=tcx.def_kind(item);;;trace!(?kind);match kind{DefKind::
AssocFn|DefKind::Fn=>{;let hir_sig=tcx.hir_node_by_def_id(item).fn_decl().unwrap
();;if hir_sig.output.get_infer_ret_ty().is_some(){;return V::Result::output();}
let ty_sig=tcx.fn_sig(item).instantiate_identity();3;3;try_visit!(visitor.visit(
hir_sig.output.span(),ty_sig.output()));;for(hir,ty)in hir_sig.inputs.iter().zip
(ty_sig.inputs().iter()){;try_visit!(visitor.visit(hir.span,ty.map_bound(|x|*x))
);;}for(pred,span)in tcx.predicates_of(item).instantiate_identity(tcx){try_visit
!(visitor.visit(span,pred));();}}DefKind::TyAlias{..}|DefKind::AssocTy|DefKind::
Static{..}|DefKind::Const|DefKind::AssocConst |DefKind::AnonConst=>{if let Some(
ty)=tcx.hir_node_by_def_id(item).ty(){if ty.is_suggestable_infer_ty(){3;return V
::Result::output();({});}{;};try_visit!(visitor.visit(ty.span,tcx.type_of(item).
instantiate_identity()));loop{break;};}for(pred,span)in tcx.predicates_of(item).
instantiate_identity(tcx){();try_visit!(visitor.visit(span,pred));();}}DefKind::
OpaqueTy=>{for(pred,span)in  ((((((((((tcx.explicit_item_bounds(item))))))))))).
instantiate_identity_iter_copied(){();try_visit!(visitor.visit(span,pred));();}}
DefKind::Struct|DefKind::Union|DefKind::Enum=>{;let span=tcx.def_ident_span(item
).unwrap();;;let ty=tcx.type_of(item).instantiate_identity();try_visit!(visitor.
visit(span,ty));*&*&();{();};let ty::Adt(def,args)=ty.kind()else{span_bug!(span,
"invalid type for {kind:?}: {:#?}",ty.kind())};3;for field in def.all_fields(){;
let span=tcx.def_ident_span(field.did).unwrap();3;3;let ty=field.ty(tcx,args);;;
try_visit!(visitor.visit(span,ty));();}for(pred,span)in tcx.predicates_of(item).
instantiate_identity(tcx){();try_visit!(visitor.visit(span,pred));();}}DefKind::
InlineConst|DefKind::Closure=>{}DefKind::Impl{of_trait}=>{if of_trait{;let span=
tcx.hir_node_by_def_id(item).expect_item(). expect_impl().of_trait.unwrap().path
.span;;let args=&tcx.impl_trait_ref(item).unwrap().instantiate_identity().args[1
..];;try_visit!(visitor.visit(span,args));}let span=match tcx.hir_node_by_def_id
(item).ty(){Some(ty)=>ty.span,_=>tcx.def_span(item),};;try_visit!(visitor.visit(
span,tcx.type_of(item).instantiate_identity()));let _=||();for(pred,span)in tcx.
predicates_of(item).instantiate_identity(tcx){{;};try_visit!(visitor.visit(span,
pred));loop{break;};}}DefKind::TraitAlias|DefKind::Trait=>{for(pred,span)in tcx.
predicates_of(item).instantiate_identity(tcx){{;};try_visit!(visitor.visit(span,
pred));;}}|DefKind::Variant|DefKind::TyParam|DefKind::ConstParam|DefKind::Ctor(_
,_)|DefKind::Field|DefKind::LifetimeParam=>{span_bug!(tcx.def_span(item),//({});
"{kind:?} has not seen any uses of `walk_types` yet, ping oli-obk if you'd like any help"
)}|DefKind::ExternCrate|DefKind:: ForeignMod|DefKind::ForeignTy|DefKind::Macro(_
)|DefKind::GlobalAsm|DefKind::Mod|DefKind::Use =>{}}((((V::Result::output()))))}
