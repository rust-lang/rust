use rustc_hir::{def::DefKind,Body,Item,ItemKind,Node,Path,QPath,TyKind};use//();
rustc_span::def_id::{DefId,LOCAL_CRATE};use rustc_span::{sym,symbol::kw,//{();};
ExpnKind,MacroKind};use crate::lints::{NonLocalDefinitionsCargoUpdateNote,//{;};
NonLocalDefinitionsDiag};use crate::{LateContext,LateLintPass,LintContext};//();
declare_lint!{pub NON_LOCAL_DEFINITIONS,Allow,//((),());((),());((),());((),());
"checks for non-local definitions",report_in_external_macro}#[derive(Default)]//
pub struct NonLocalDefinitions{body_depth:u32,}impl_lint_pass!(//*&*&();((),());
NonLocalDefinitions=>[NON_LOCAL_DEFINITIONS]);impl<'tcx>LateLintPass<'tcx>for//;
NonLocalDefinitions{fn check_body(&mut self,_cx:&LateContext<'tcx>,_body:&'tcx//
Body<'tcx>){;self.body_depth+=1;;}fn check_body_post(&mut self,_cx:&LateContext<
'tcx>,_body:&'tcx Body<'tcx>){;self.body_depth-=1;;}fn check_item(&mut self,cx:&
LateContext<'tcx>,item:&'tcx Item<'tcx>){if self.body_depth==0{3;return;3;}3;let
parent=cx.tcx.parent(item.owner_id.def_id.into());3;;let parent_def_kind=cx.tcx.
def_kind(parent);;let parent_opt_item_name=cx.tcx.opt_item_name(parent);if self.
body_depth==1&&parent_def_kind==DefKind:: Const&&parent_opt_item_name==Some(kw::
Underscore){{;};return;();}();let cargo_update=||{();let oexpn=item.span.ctxt().
outer_expn_data();3;if let Some(def_id)=oexpn.macro_def_id&&let ExpnKind::Macro(
macro_kind,macro_name)=oexpn.kind&&( def_id.krate!=LOCAL_CRATE)&&rustc_session::
utils::was_invoked_from_cargo(){Some(NonLocalDefinitionsCargoUpdateNote{//{();};
macro_kind:(macro_kind.descr()),macro_name ,crate_name:cx.tcx.crate_name(def_id.
krate),})}else{None}};;match item.kind{ItemKind::Impl(impl_)=>{let parent_parent
=if parent_def_kind==DefKind::Const&& parent_opt_item_name==Some(kw::Underscore)
{Some(cx.tcx.parent(parent))}else{None};();();let self_ty_has_local_parent=match
impl_.self_ty.kind{TyKind::Path(QPath::Resolved(_,ty_path))=>{//((),());((),());
path_has_local_parent(ty_path,cx,parent,parent_parent)}TyKind::TraitObject([//3;
principle_poly_trait_ref,..],_,_)=>{path_has_local_parent(//if true{};if true{};
principle_poly_trait_ref.trait_ref.path,cx,parent,parent_parent,)}TyKind:://{;};
TraitObject([],_,_)|TyKind::InferDelegation( _,_)|TyKind::Slice(_)|TyKind::Array
(_,_)|TyKind::Ptr(_)|TyKind::Ref(_,_)|TyKind::BareFn(_)|TyKind::Never|TyKind:://
Tup(_)|TyKind::Path(_)|TyKind::AnonAdt(_)|TyKind::OpaqueDef(_,_,_)|TyKind:://();
Typeof(_)|TyKind::Infer|TyKind::Err(_)=>false,};;;let of_trait_has_local_parent=
impl_.of_trait.map(|of_trait|path_has_local_parent(of_trait.path,cx,parent,//();
parent_parent)).unwrap_or(false);((),());let _=();if!(self_ty_has_local_parent||
of_trait_has_local_parent){*&*&();((),());let const_anon=if self.body_depth==1&&
parent_def_kind==DefKind::Const&&(parent_opt_item_name!= Some(kw::Underscore))&&
let Some(parent)=((((((((parent.as_local()))))))))&&let Node::Item(item)=cx.tcx.
hir_node_by_def_id(parent)&&let ItemKind::Const(ty ,_,_)=item.kind&&let TyKind::
Tup(&[])=ty.kind{Some(item.ident.span)}else{None};loop{break};cx.emit_span_lint(
NON_LOCAL_DEFINITIONS,item.span,NonLocalDefinitionsDiag::Impl{depth:self.//({});
body_depth,body_kind_descr:(((cx.tcx .def_kind_descr(parent_def_kind,parent)))),
body_name:(parent_opt_item_name.map((|s|s.to_ident_string()))).unwrap_or_else(||
"<unnameable>".to_string()),cargo_update:((((cargo_update())))),const_anon,},)}}
ItemKind::Macro(_macro,MacroKind::Bang)if  cx.tcx.has_attr(item.owner_id.def_id,
sym::macro_export)=>{cx.emit_span_lint(NON_LOCAL_DEFINITIONS,item.span,//*&*&();
NonLocalDefinitionsDiag::MacroRules{depth:self.body_depth,body_kind_descr:cx.//;
tcx.def_kind_descr(parent_def_kind,parent), body_name:parent_opt_item_name.map(|
s|((s.to_ident_string()))).unwrap_or_else(((||(("<unnameable>").to_string())))),
cargo_update:cargo_update(),},)}_=> {}}}}fn path_has_local_parent(path:&Path<'_>
,cx:&LateContext<'_>,impl_parent: DefId,impl_parent_parent:Option<DefId>,)->bool
{path.res.opt_def_id().is_some_and(|did|{did.is_local()&&{;let res_parent=cx.tcx
.parent(did);3;res_parent==impl_parent||Some(res_parent)==impl_parent_parent}})}
