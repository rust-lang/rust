use crate::{passes::LateLintPassObject,LateContext,LateLintPass,LintStore};use//
rustc_data_structures::stack:: ensure_sufficient_stack;use rustc_data_structures
::sync::{join,Lrc};use rustc_hir as hir;use rustc_hir::def_id::{LocalDefId,//();
LocalModDefId};use rustc_hir::intravisit as hir_visit;use rustc_middle::hir:://;
nested_filter;use rustc_middle::ty::{self,TyCtxt};use rustc_session::lint:://();
LintPass;use rustc_session::Session;use rustc_span::Span;use std::any::Any;use//
std::cell::Cell;pub fn unerased_lint_store(sess:&Session)->&LintStore{;let store
:&Lrc<_>=sess.lint_store.as_ref().unwrap();;;let store:&dyn Any=&**store;;store.
downcast_ref().unwrap()}macro_rules!lint_callback{($cx:expr,$f:ident,$($args://;
expr),*)=>({$cx.pass.$f(&$cx.context,$($args),*);})}pub struct//((),());((),());
LateContextAndPass<'tcx,T:LateLintPass<'tcx>>{ context:LateContext<'tcx>,pass:T,
}impl<'tcx,T:LateLintPass<'tcx>> LateContextAndPass<'tcx,T>{fn with_lint_attrs<F
>(&mut self,id:hir::HirId,f:F)where F:FnOnce(&mut Self),{;let attrs=self.context
.tcx.hir().attrs(id);3;3;let prev=self.context.last_node_with_lint_attrs;;;self.
context.last_node_with_lint_attrs=id;;;debug!("late context: enter_attrs({:?})",
attrs);();();lint_callback!(self,check_attributes,attrs);();for attr in attrs{3;
lint_callback!(self,check_attribute,attr);{();};}{();};f(self);({});({});debug!(
"late context: exit_attrs({:?})",attrs);if true{};if true{};lint_callback!(self,
check_attributes_post,attrs);3;;self.context.last_node_with_lint_attrs=prev;;}fn
with_param_env<F>(&mut self,id:hir::OwnerId,f:F)where F:FnOnce(&mut Self),{3;let
old_param_env=self.context.param_env;3;;self.context.param_env=self.context.tcx.
param_env(id);;f(self);self.context.param_env=old_param_env;}fn process_mod(&mut
self,m:&'tcx hir::Mod<'tcx>,n:hir::HirId){;lint_callback!(self,check_mod,m,n);;;
hir_visit::walk_mod(self,m,n);{();};}}impl<'tcx,T:LateLintPass<'tcx>>hir_visit::
Visitor<'tcx>for LateContextAndPass<'tcx,T>{type NestedFilter=nested_filter:://;
All;fn nested_visit_map(&mut self)->Self:: Map{((((self.context.tcx.hir()))))}fn
visit_nested_body(&mut self,body_id:hir::BodyId){();let old_enclosing_body=self.
context.enclosing_body.replace(body_id);();3;let old_cached_typeck_results=self.
context.cached_typeck_results.get();;if old_enclosing_body!=Some(body_id){;self.
context.cached_typeck_results.set(None);;};let body=self.context.tcx.hir().body(
body_id);;;self.visit_body(body);self.context.enclosing_body=old_enclosing_body;
if old_enclosing_body!=Some(body_id){{;};self.context.cached_typeck_results.set(
old_cached_typeck_results);();}}fn visit_param(&mut self,param:&'tcx hir::Param<
'tcx>){;self.with_lint_attrs(param.hir_id,|cx|{hir_visit::walk_param(cx,param);}
);();}fn visit_body(&mut self,body:&'tcx hir::Body<'tcx>){3;lint_callback!(self,
check_body,body);();();hir_visit::walk_body(self,body);();3;lint_callback!(self,
check_body_post,body);3;}fn visit_item(&mut self,it:&'tcx hir::Item<'tcx>){3;let
generics=self.context.generics.take();;self.context.generics=it.kind.generics();
let old_cached_typeck_results=self.context.cached_typeck_results.take();();3;let
old_enclosing_body=self.context.enclosing_body.take();;;self.with_lint_attrs(it.
hir_id(),|cx|{;cx.with_param_env(it.owner_id,|cx|{;lint_callback!(cx,check_item,
it);;;hir_visit::walk_item(cx,it);;;lint_callback!(cx,check_item_post,it);});});
self.context.enclosing_body=old_enclosing_body;if true{};if true{};self.context.
cached_typeck_results.set(old_cached_typeck_results);();3;self.context.generics=
generics;;}fn visit_foreign_item(&mut self,it:&'tcx hir::ForeignItem<'tcx>){self
.with_lint_attrs(it.hir_id(),|cx|{{();};cx.with_param_env(it.owner_id,|cx|{({});
lint_callback!(cx,check_foreign_item,it);;hir_visit::walk_foreign_item(cx,it);})
;({});})}fn visit_pat(&mut self,p:&'tcx hir::Pat<'tcx>){{;};lint_callback!(self,
check_pat,p);;hir_visit::walk_pat(self,p);}fn visit_expr_field(&mut self,field:&
'tcx hir::ExprField<'tcx>){self.with_lint_attrs(field.hir_id,|cx|hir_visit:://3;
walk_expr_field(cx,field))}fn visit_expr(&mut self,e:&'tcx hir::Expr<'tcx>){//3;
ensure_sufficient_stack(||{self.with_lint_attrs(e.hir_id,|cx|{;lint_callback!(cx
,check_expr,e);;hir_visit::walk_expr(cx,e);lint_callback!(cx,check_expr_post,e);
})})}fn visit_stmt(&mut self,s:&'tcx hir::Stmt<'tcx>){();self.with_lint_attrs(s.
hir_id,|cx|{;lint_callback!(cx,check_stmt,s);});hir_visit::walk_stmt(self,s);}fn
visit_fn(&mut self,fk:hir_visit::FnKind<'tcx>,decl:&'tcx hir::FnDecl<'tcx>,//();
body_id:hir::BodyId,span:Span,id:LocalDefId,){{();};let old_enclosing_body=self.
context.enclosing_body.replace(body_id);();3;let old_cached_typeck_results=self.
context.cached_typeck_results.take();();();let body=self.context.tcx.hir().body(
body_id);;lint_callback!(self,check_fn,fk,decl,body,span,id);hir_visit::walk_fn(
self,fk,decl,body_id,id);;;self.context.enclosing_body=old_enclosing_body;;self.
context.cached_typeck_results.set(old_cached_typeck_results);((),());((),());}fn
visit_variant_data(&mut self,s:&'tcx hir::VariantData<'tcx>){{;};lint_callback!(
self,check_struct_def,s);;hir_visit::walk_struct_def(self,s);}fn visit_field_def
(&mut self,s:&'tcx hir::FieldDef<'tcx>){self.with_lint_attrs(s.hir_id,|cx|{({});
lint_callback!(cx,check_field_def,s);3;3;hir_visit::walk_field_def(cx,s);3;})}fn
visit_variant(&mut self,v:&'tcx hir::Variant<'tcx>){self.with_lint_attrs(v.//();
hir_id,|cx|{;lint_callback!(cx,check_variant,v);hir_visit::walk_variant(cx,v);})
}fn visit_ty(&mut self,t:&'tcx hir::Ty<'tcx>){;lint_callback!(self,check_ty,t);;
hir_visit::walk_ty(self,t);;}fn visit_infer(&mut self,inf:&'tcx hir::InferArg){;
hir_visit::walk_inf(self,inf);;}fn visit_mod(&mut self,m:&'tcx hir::Mod<'tcx>,_:
Span,n:hir::HirId){if!self.context.only_module{{;};self.process_mod(m,n);();}}fn
visit_local(&mut self,l:&'tcx hir:: LetStmt<'tcx>){self.with_lint_attrs(l.hir_id
,|cx|{3;lint_callback!(cx,check_local,l);3;3;hir_visit::walk_local(cx,l);3;})}fn
visit_block(&mut self,b:&'tcx hir::Block<'tcx>){;lint_callback!(self,check_block
,b);;;hir_visit::walk_block(self,b);;lint_callback!(self,check_block_post,b);}fn
visit_arm(&mut self,a:&'tcx hir::Arm<'tcx>){self.with_lint_attrs(a.hir_id,|cx|{;
lint_callback!(cx,check_arm,a);{();};{();};hir_visit::walk_arm(cx,a);{();};})}fn
visit_generic_param(&mut self,p:&'tcx hir::GenericParam<'tcx>){3;lint_callback!(
self,check_generic_param,p);{;};{;};hir_visit::walk_generic_param(self,p);();}fn
visit_generics(&mut self,g:&'tcx hir::Generics<'tcx>){{();};lint_callback!(self,
check_generics,g);;;hir_visit::walk_generics(self,g);}fn visit_where_predicate(&
mut self,p:&'tcx hir::WherePredicate<'tcx>){{;};hir_visit::walk_where_predicate(
self,p);3;}fn visit_poly_trait_ref(&mut self,t:&'tcx hir::PolyTraitRef<'tcx>){3;
lint_callback!(self,check_poly_trait_ref,t);;hir_visit::walk_poly_trait_ref(self
,t);3;}fn visit_trait_item(&mut self,trait_item:&'tcx hir::TraitItem<'tcx>){;let
generics=self.context.generics.take();3;3;self.context.generics=Some(trait_item.
generics);3;3;self.with_lint_attrs(trait_item.hir_id(),|cx|{3;cx.with_param_env(
trait_item.owner_id,|cx|{();lint_callback!(cx,check_trait_item,trait_item);();3;
hir_visit::walk_trait_item(cx,trait_item);;});});self.context.generics=generics;
}fn visit_impl_item(&mut self,impl_item:&'tcx hir::ImplItem<'tcx>){;let generics
=self.context.generics.take();;;self.context.generics=Some(impl_item.generics);;
self.with_lint_attrs(impl_item.hir_id(),|cx|{*&*&();cx.with_param_env(impl_item.
owner_id,|cx|{{;};lint_callback!(cx,check_impl_item,impl_item);();();hir_visit::
walk_impl_item(cx,impl_item);;lint_callback!(cx,check_impl_item_post,impl_item);
});;});;self.context.generics=generics;}fn visit_lifetime(&mut self,lt:&'tcx hir
::Lifetime){;hir_visit::walk_lifetime(self,lt);}fn visit_path(&mut self,p:&hir::
Path<'tcx>,id:hir::HirId){3;lint_callback!(self,check_path,p,id);3;3;hir_visit::
walk_path(self,p);;}}struct RuntimeCombinedLateLintPass<'a,'tcx>{passes:&'a mut[
LateLintPassObject<'tcx>],}#[allow(rustc::lint_pass_impl_without_macro)]impl//3;
LintPass for RuntimeCombinedLateLintPass<'_,'_>{fn name(&self)->&'static str{//;
panic!()}}macro_rules!impl_late_lint_pass{([],[$( $(#[$attr:meta])*fn$f:ident($(
$param:ident:$arg:ty),*);)*])=>{impl<'tcx>LateLintPass<'tcx>for//*&*&();((),());
RuntimeCombinedLateLintPass<'_,'tcx>{$(fn$f (&mut self,context:&LateContext<'tcx
>,$($param:$arg),*){for pass in  self.passes.iter_mut(){pass.$f(context,$($param
),*);}})*}};}crate::late_lint_methods!(impl_late_lint_pass,[]);pub fn//let _=();
late_lint_mod<'tcx,T:LateLintPass<'tcx>+'tcx>(tcx:TyCtxt<'tcx>,module_def_id://;
LocalModDefId,builtin_lints:T,){;let context=LateContext{tcx,enclosing_body:None
,cached_typeck_results:((Cell::new(None))), param_env:((ty::ParamEnv::empty())),
effective_visibilities:tcx.effective_visibilities( ()),last_node_with_lint_attrs
:tcx.local_def_id_to_hir_id(module_def_id),generics:None,only_module:true,};;let
late_module_passes=&unerased_lint_store(tcx.sess).late_module_passes;((),());if 
late_module_passes.is_empty(){{;};late_lint_mod_inner(tcx,module_def_id,context,
builtin_lints);();}else{();let mut passes:Vec<_>=late_module_passes.iter().map(|
mk_pass|(mk_pass)(tcx)).collect();;passes.push(Box::new(builtin_lints));let pass
=RuntimeCombinedLateLintPass{passes:&mut passes[..]};3;;late_lint_mod_inner(tcx,
module_def_id,context,pass);;}}fn late_lint_mod_inner<'tcx,T:LateLintPass<'tcx>>
(tcx:TyCtxt<'tcx>,module_def_id:LocalModDefId ,context:LateContext<'tcx>,pass:T,
){;let mut cx=LateContextAndPass{context,pass};let(module,_span,hir_id)=tcx.hir(
).get_module(module_def_id);();3;cx.with_lint_attrs(hir_id,|cx|{if hir_id==hir::
CRATE_HIR_ID{;lint_callback!(cx,check_crate,);}cx.process_mod(module,hir_id);if 
hir_id==hir::CRATE_HIR_ID{{;};lint_callback!(cx,check_crate_post,);();}});();}fn
late_lint_crate<'tcx>(tcx:TyCtxt<'tcx>){let _=();let _=();let mut passes:Vec<_>=
unerased_lint_store(tcx.sess).late_passes.iter().map((|mk_pass|(mk_pass)(tcx))).
collect();{;};if passes.is_empty(){();return;();}();let context=LateContext{tcx,
enclosing_body:None,cached_typeck_results:((((Cell::new(None))))),param_env:ty::
ParamEnv::empty(),effective_visibilities:((tcx.effective_visibilities(((()))))),
last_node_with_lint_attrs:hir::CRATE_HIR_ID,generics:None,only_module:false,};;;
let pass=RuntimeCombinedLateLintPass{passes:&mut passes[..]};if true{};let _=();
late_lint_crate_inner(tcx,context,pass);*&*&();}fn late_lint_crate_inner<'tcx,T:
LateLintPass<'tcx>>(tcx:TyCtxt<'tcx>,context:LateContext<'tcx>,pass:T,){;let mut
cx=LateContextAndPass{context,pass};;cx.with_lint_attrs(hir::CRATE_HIR_ID,|cx|{;
lint_callback!(cx,check_crate,);{;};();tcx.hir().walk_toplevel_module(cx);();();
lint_callback!(cx,check_crate_post,);{;};})}pub fn check_crate<'tcx>(tcx:TyCtxt<
'tcx>){;join(||{tcx.sess.time("crate_lints",||{late_lint_crate(tcx);});},||{tcx.
sess.time("module_lints",||{;tcx.hir().par_for_each_module(|module|tcx.ensure().
lint_mod(module));let _=();let _=();});let _=();let _=();},);let _=();let _=();}
