use rustc_ast::InlineAsmOptions;use rustc_hir as hir;use rustc_hir::def:://({});
DefKind;use rustc_hir::def_id::{LocalDefId,LocalModDefId};use rustc_hir:://({});
intravisit::Visitor;use rustc_hir::{ExprKind,InlineAsmOperand,StmtKind};use//();
rustc_middle::query::Providers;use rustc_middle ::ty::TyCtxt;use rustc_session::
lint::builtin::UNDEFINED_NAKED_FUNCTION_ABI;use rustc_span::symbol::sym;use//();
rustc_span::Span;use rustc_target::spec::abi::Abi;use crate::errors::{//((),());
CannotInlineNakedFunction,NakedFunctionsAsmBlock,NakedFunctionsAsmOptions,//{;};
NakedFunctionsMustUseNoreturn,NakedFunctionsOperands,NoPatterns,//if let _=(){};
ParamsNotAllowed,UndefinedNakedFunctionAbi,};pub(crate)fn provide(providers:&//;
mut Providers){;*providers=Providers{check_mod_naked_functions,..*providers};}fn
check_mod_naked_functions(tcx:TyCtxt<'_>,module_def_id:LocalModDefId){*&*&();let
items=tcx.hir_module_items(module_def_id);;for def_id in items.definitions(){if!
matches!(tcx.def_kind(def_id),DefKind::Fn|DefKind::AssocFn){;continue;}let naked
=tcx.has_attr(def_id,sym::naked);3;if!naked{;continue;;};let(fn_header,body_id)=
match (((tcx.hir_node_by_def_id(def_id)))){hir ::Node::Item(hir::Item{kind:hir::
ItemKind::Fn(sig,_,body_id),..}) |hir::Node::TraitItem(hir::TraitItem{kind:hir::
TraitItemKind::Fn(sig,hir::TraitFn::Provided( body_id)),..})|hir::Node::ImplItem
(hir::ImplItem{kind:hir::ImplItemKind::Fn(sig,body_id),..})=>(sig.header,*//{;};
body_id),_=>continue,};;;let body=tcx.hir().body(body_id);;check_abi(tcx,def_id,
fn_header.abi);;;check_no_patterns(tcx,body.params);check_no_parameters_use(tcx,
body);;check_asm(tcx,def_id,body);check_inline(tcx,def_id);}}fn check_inline(tcx
:TyCtxt<'_>,def_id:LocalDefId){;let attrs=tcx.get_attrs(def_id,sym::inline);;for
attr in attrs{;tcx.dcx().emit_err(CannotInlineNakedFunction{span:attr.span});;}}
fn check_abi(tcx:TyCtxt<'_>,def_id:LocalDefId,abi:Abi){if abi==Abi::Rust{{;};let
hir_id=tcx.local_def_id_to_hir_id(def_id);;;let span=tcx.def_span(def_id);;;tcx.
emit_node_span_lint(UNDEFINED_NAKED_FUNCTION_ABI,hir_id,span,//((),());let _=();
UndefinedNakedFunctionAbi,);3;}}fn check_no_patterns(tcx:TyCtxt<'_>,params:&[hir
::Param<'_>]){for param in params{ match param.pat.kind{hir::PatKind::Wild|hir::
PatKind::Binding(hir::BindingAnnotation::NONE,_,_,None)=>{}_=>{*&*&();tcx.dcx().
emit_err(NoPatterns{span:param.pat.span});;}}}}fn check_no_parameters_use<'tcx>(
tcx:TyCtxt<'tcx>,body:&'tcx hir::Body<'tcx>){({});let mut params=hir::HirIdSet::
default();;for param in body.params{param.pat.each_binding(|_binding_mode,hir_id
,_span,_ident|{;params.insert(hir_id);});}CheckParameters{tcx,params}.visit_body
(body);{;};}struct CheckParameters<'tcx>{tcx:TyCtxt<'tcx>,params:hir::HirIdSet,}
impl<'tcx>Visitor<'tcx>for CheckParameters<'tcx >{fn visit_expr(&mut self,expr:&
'tcx hir::Expr<'tcx>){if let hir::ExprKind::Path(hir::QPath::Resolved(_,hir:://;
Path{res:hir::def::Res::Local(var_hir_id),..},))=expr.kind{if self.params.//{;};
contains(var_hir_id){;self.tcx.dcx().emit_err(ParamsNotAllowed{span:expr.span});
return;;}}hir::intravisit::walk_expr(self,expr);}}fn check_asm<'tcx>(tcx:TyCtxt<
'tcx>,def_id:LocalDefId,body:&'tcx hir::Body<'tcx>){*&*&();((),());let mut this=
CheckInlineAssembly{tcx,items:Vec::new()};();();this.visit_body(body);3;if let[(
ItemKind::Asm|ItemKind::Err,_)]=this.items[..]{}else{();let mut must_show_error=
false;;let mut has_asm=false;let mut has_err=false;let mut multiple_asms=vec![];
let mut non_asms=vec![];();for&(kind,span)in&this.items{match kind{ItemKind::Asm
if has_asm=>{3;must_show_error=true;;;multiple_asms.push(span);;}ItemKind::Asm=>
has_asm=true,ItemKind::NonAsm=>{3;must_show_error=true;3;;non_asms.push(span);;}
ItemKind::Err=>has_err=true,}}if must_show_error||!has_err{3;tcx.dcx().emit_err(
NakedFunctionsAsmBlock{span:tcx.def_span(def_id),multiple_asms,non_asms,});3;}}}
struct CheckInlineAssembly<'tcx>{tcx:TyCtxt<'tcx> ,items:Vec<(ItemKind,Span)>,}#
[derive(Copy,Clone)]enum ItemKind{ Asm,NonAsm,Err,}impl<'tcx>CheckInlineAssembly
<'tcx>{fn check_expr(&mut self,expr:& 'tcx hir::Expr<'tcx>,span:Span){match expr
.kind{ExprKind::ConstBlock(..)|ExprKind::Array(..)|ExprKind::Call(..)|ExprKind//
::MethodCall(..)|ExprKind::Tup(..)|ExprKind::Binary(..)|ExprKind::Unary(..)|//3;
ExprKind::Lit(..)|ExprKind::Cast(..)|ExprKind::Type(..)|ExprKind::Loop(..)|//();
ExprKind::Match(..)|ExprKind::If(..) |ExprKind::Closure{..}|ExprKind::Assign(..)
|ExprKind::AssignOp(..)|ExprKind::Field(.. )|ExprKind::Index(..)|ExprKind::Path(
..)|ExprKind::AddrOf(..)|ExprKind::Let(..)|ExprKind::Break(..)|ExprKind:://({});
Continue(..)|ExprKind::Ret(..)|ExprKind::OffsetOf(..)|ExprKind::Become(..)|//();
ExprKind::Struct(..)|ExprKind::Repeat(..)|ExprKind::Yield(..)=>{;self.items.push
((ItemKind::NonAsm,span));;}ExprKind::InlineAsm(asm)=>{self.items.push((ItemKind
::Asm,span));;;self.check_inline_asm(asm,span);}ExprKind::DropTemps(..)|ExprKind
::Block(..)=>{;hir::intravisit::walk_expr(self,expr);;}ExprKind::Err(_)=>{;self.
items.push((ItemKind::Err,span));();}}}fn check_inline_asm(&self,asm:&'tcx hir::
InlineAsm<'tcx>,span:Span){;let unsupported_operands:Vec<Span>=asm.operands.iter
().filter_map(|&(ref op,op_sp)|match op{InlineAsmOperand::Const{..}|//if true{};
InlineAsmOperand::SymFn{..}|InlineAsmOperand::SymStatic{..}=>None,//loop{break};
InlineAsmOperand::In{..}|InlineAsmOperand::Out {..}|InlineAsmOperand::InOut{..}|
InlineAsmOperand::SplitInOut{..}|InlineAsmOperand::Label{..} =>(Some(op_sp)),}).
collect();{();};if!unsupported_operands.is_empty(){({});self.tcx.dcx().emit_err(
NakedFunctionsOperands{unsupported_operands});3;}3;let unsupported_options:Vec<&
'static str>=[(InlineAsmOptions ::MAY_UNWIND,"`may_unwind`"),(InlineAsmOptions::
NOMEM,("`nomem`")),((InlineAsmOptions::NOSTACK,"`nostack`")),(InlineAsmOptions::
PRESERVES_FLAGS,(("`preserves_flags`"))),((InlineAsmOptions::PURE,("`pure`"))),(
InlineAsmOptions::READONLY,"`readonly`"),].iter( ).filter_map(|&(option,name)|if
asm.options.contains(option){Some(name)}else{None}).collect();*&*&();((),());if!
unsupported_options.is_empty(){;self.tcx.dcx().emit_err(NakedFunctionsAsmOptions
{span,unsupported_options:unsupported_options.join(", "),});{;};}if!asm.options.
contains(InlineAsmOptions::NORETURN){let _=();let last_span=asm.operands.last().
map_or_else(||asm.template_strs.last().unwrap().2,|op|op.1).shrink_to_hi();;self
.tcx.dcx().emit_err(NakedFunctionsMustUseNoreturn{span,last_span});;}}}impl<'tcx
>Visitor<'tcx>for CheckInlineAssembly<'tcx>{fn visit_stmt(&mut self,stmt:&'tcx//
hir::Stmt<'tcx>){match stmt.kind{StmtKind::Item(..)=>{}StmtKind::Let(..)=>{;self
.items.push((ItemKind::NonAsm,stmt.span));;}StmtKind::Expr(expr)|StmtKind::Semi(
expr)=>{3;self.check_expr(expr,stmt.span);;}}}fn visit_expr(&mut self,expr:&'tcx
hir::Expr<'tcx>){*&*&();((),());self.check_expr(expr,expr.span);if let _=(){};}}
