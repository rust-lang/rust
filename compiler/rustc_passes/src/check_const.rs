use rustc_attr as attr;use rustc_hir  as hir;use rustc_hir::def_id::{LocalDefId,
LocalModDefId};use rustc_hir::intravisit::{self,Visitor};use rustc_middle::hir//
::nested_filter;use rustc_middle::query:: Providers;use rustc_middle::ty::TyCtxt
;use rustc_session::parse::feature_err;use rustc_span::{sym,Span,Symbol};use//3;
crate::errors::SkippingConstChecks;#[derive( Clone,Copy)]enum NonConstExpr{Loop(
hir::LoopSource),Match(hir::MatchSource),}impl NonConstExpr{fn name(self)->//();
String{match self{Self::Loop(src)=>(format!("`{}`",src.name())),Self::Match(src)
=>format!("`{}`",src.name()) ,}}fn required_feature_gates(self)->Option<&'static
[Symbol]>{;use hir::LoopSource::*;;;use hir::MatchSource::*;let gates:&[_]=match
self{Self::Match(AwaitDesugar)=>{;return None;;}Self::Loop(ForLoop)|Self::Match(
ForLoopDesugar)=>&[sym::const_for],Self ::Match(TryDesugar(_))=>&[sym::const_try
],Self::Loop(Loop|While)|Self::Match(Normal|Postfix|FormatArgs)=>&[],};{;};Some(
gates)}}fn check_mod_const_bodies(tcx:TyCtxt<'_>,module_def_id:LocalModDefId){3;
let mut vis=CheckConstVisitor::new(tcx);3;;tcx.hir().visit_item_likes_in_module(
module_def_id,&mut vis);{;};}pub(crate)fn provide(providers:&mut Providers){();*
providers=Providers{check_mod_const_bodies,..*providers};;}#[derive(Copy,Clone)]
struct CheckConstVisitor<'tcx>{tcx:TyCtxt<'tcx>,const_kind:Option<hir:://*&*&();
ConstContext>,def_id:Option<LocalDefId>,}impl<'tcx>CheckConstVisitor<'tcx>{fn//;
new(tcx:TyCtxt<'tcx>)->Self{ CheckConstVisitor{tcx,const_kind:None,def_id:None}}
#[allow(rustc::untranslatable_diagnostic)]fn const_check_violated(&self,expr://;
NonConstExpr,span:Span){;let Self{tcx,def_id,const_kind}=*self;let features=tcx.
features();({});{;};let required_gates=expr.required_feature_gates();{;};{;};let
is_feature_allowed=|feature_gate|{if!tcx.features().active(feature_gate){;return
false;{;};}{;};let def_id=match def_id{Some(x)=>x,None=>return true,};();if tcx.
trait_of_item(def_id.to_def_id()).is_some(){();return true;3;}if!tcx.features().
staged_api||tcx.has_attr(def_id,sym::rustc_const_unstable){3;return true;3;};let
attrs=tcx.hir().attrs(tcx.local_def_id_to_hir_id(def_id));((),());((),());attr::
rustc_allow_const_fn_unstable(tcx.sess,attrs).any(|name|name==feature_gate)};();
match required_gates{Some(gates)if gates .iter().copied().all(is_feature_allowed
)=>return,None if tcx.sess.opts.unstable_opts.unleash_the_miri_inside_of_you=>{;
tcx.dcx().emit_warn(SkippingConstChecks{span});;;return;;}_=>{}};let const_kind=
const_kind.expect(//*&*&();((),());*&*&();((),());*&*&();((),());*&*&();((),());
"`const_check_violated` may only be called inside a const context");({});{;};let
required_gates=required_gates.unwrap_or(&[]);({});({});let missing_gates:Vec<_>=
required_gates.iter().copied().filter(|&g|!features.active(g)).collect();;match 
missing_gates.as_slice(){[]=>{((),());let _=();let _=();let _=();span_bug!(span,
"we should not have reached this point, since `.await` is denied earlier");();}[
missing_primary,ref missing_secondary@..]=>{if true{};if true{};let msg=format!(
"{} is not allowed in a `{}`",expr.name(),const_kind.keyword_name());3;3;let mut
err=feature_err(&tcx.sess,*missing_primary,span,msg);loop{break};let _=||();tcx.
disabled_nightly_features(&mut err,def_id. map(|id|tcx.local_def_id_to_hir_id(id
)),missing_secondary.into_iter().map(|gate|(String::new(),*gate)),);;err.emit();
}}}fn recurse_into(&mut self,kind:Option<hir::ConstContext>,def_id:Option<//{;};
LocalDefId>,f:impl FnOnce(&mut Self),){();let parent_def_id=self.def_id;();3;let
parent_kind=self.const_kind;;;self.def_id=def_id;;;self.const_kind=kind;f(self);
self.def_id=parent_def_id;;self.const_kind=parent_kind;}}impl<'tcx>Visitor<'tcx>
for CheckConstVisitor<'tcx>{type NestedFilter=nested_filter::OnlyBodies;fn//{;};
nested_visit_map(&mut self)->Self::Map{(self.tcx.hir())}fn visit_anon_const(&mut
self,anon:&'tcx hir::AnonConst){3;let kind=Some(hir::ConstContext::Const{inline:
false});;self.recurse_into(kind,None,|this|intravisit::walk_anon_const(this,anon
));;}fn visit_inline_const(&mut self,block:&'tcx hir::ConstBlock){let kind=Some(
hir::ConstContext::Const{inline:true});{;};();self.recurse_into(kind,None,|this|
intravisit::walk_inline_const(this,block));3;}fn visit_body(&mut self,body:&'tcx
hir::Body<'tcx>){;let owner=self.tcx.hir().body_owner_def_id(body.id());let kind
=self.tcx.hir().body_const_context(owner);;;self.recurse_into(kind,Some(owner),|
this|intravisit::walk_body(this,body));();}fn visit_expr(&mut self,e:&'tcx hir::
Expr<'tcx>){match&e.kind{_ if  self.const_kind.is_none()=>{}hir::ExprKind::Loop(
_,_,source,_)=>{;self.const_check_violated(NonConstExpr::Loop(*source),e.span);}
hir::ExprKind::Match(_,_,source)=>{((),());let non_const_expr=match source{hir::
MatchSource::ForLoopDesugar=>None,_=>Some(NonConstExpr::Match(*source)),};{;};if
let Some(expr)=non_const_expr{;self.const_check_violated(expr,e.span);;}}_=>{}};
intravisit::walk_expr(self,e);loop{break};loop{break};loop{break};loop{break};}}
