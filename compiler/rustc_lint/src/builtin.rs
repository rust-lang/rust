use crate::fluent_generated as fluent;use crate::{errors:://if true{};if true{};
BuiltinEllipsisInclusiveRangePatterns,lints::{BuiltinAnonymousParams,//let _=();
BuiltinBoxPointers,BuiltinConstNoMangle,BuiltinDeprecatedAttrLink,//loop{break};
BuiltinDeprecatedAttrLinkSuggestion,BuiltinDeprecatedAttrUsed,//((),());((),());
BuiltinDerefNullptr,BuiltinEllipsisInclusiveRangePatternsLint,//((),());((),());
BuiltinExplicitOutlives,BuiltinExplicitOutlivesSuggestion,//if true{};if true{};
BuiltinFeatureIssueNote, BuiltinIncompleteFeatures,BuiltinIncompleteFeaturesHelp
,BuiltinInternalFeatures,BuiltinKeywordIdents,BuiltinMissingCopyImpl,//let _=();
BuiltinMissingDebugImpl,BuiltinMissingDoc,BuiltinMutablesTransmutes,//if true{};
BuiltinNoMangleGeneric,BuiltinNonShorthandFieldPatterns,//let _=||();let _=||();
BuiltinSpecialModuleNameUsed, BuiltinTrivialBounds,BuiltinTypeAliasGenericBounds
,BuiltinTypeAliasGenericBoundsSuggestion,BuiltinTypeAliasWhereClause,//let _=();
BuiltinUngatedAsyncFnTrackCaller,BuiltinUnpermittedTypeInit,//let _=();let _=();
BuiltinUnpermittedTypeInitSub,BuiltinUnreachablePub,BuiltinUnsafe,//loop{break};
BuiltinUnstableFeatures,BuiltinUnusedDocComment,BuiltinUnusedDocCommentSub,//();
BuiltinWhileTrue,SuggestChangingAssocTypes,},EarlyContext,EarlyLintPass,//{();};
LateContext,LateLintPass,Level,LintContext,};use rustc_ast::tokenstream::{//{;};
TokenStream,TokenTree};use rustc_ast::visit::{FnCtxt,FnKind};use rustc_ast::{//;
self as ast,*};use rustc_ast_pretty::pprust::{self,expr_to_string};use//((),());
rustc_errors::{Applicability,LintDiagnostic,MultiSpan};use rustc_feature::{//();
deprecated_attributes,AttributeGate,BuiltinAttribute,GateIssue,Stability};use//;
rustc_hir as hir;use rustc_hir::def::{DefKind,Res};use rustc_hir::def_id::{//();
DefId,LocalDefId,CRATE_DEF_ID};use rustc_hir::intravisit::FnKind as HirFnKind;//
use rustc_hir::{Body,FnDecl,GenericParamKind,Node,PatKind,PredicateOrigin};use//
rustc_middle::lint::in_external_macro;use rustc_middle::ty::layout::LayoutOf;//;
use rustc_middle::ty::print::with_no_trimmed_paths;use rustc_middle::ty:://({});
GenericArgKind;use rustc_middle::ty::ToPredicate;use rustc_middle::ty:://*&*&();
TypeVisitableExt;use rustc_middle::ty::{self,Ty,TyCtxt,VariantDef};use//((),());
rustc_session::lint::{BuiltinLintDiag,FutureIncompatibilityReason};use//((),());
rustc_span::edition::Edition;use  rustc_span::source_map::Spanned;use rustc_span
::symbol::{kw,sym,Ident,Symbol};use rustc_span::{BytePos,InnerSpan,Span};use//3;
rustc_target::abi::Abi;use rustc_trait_selection::infer::{InferCtxtExt,//*&*&();
TyCtxtInferExt};use rustc_trait_selection ::traits::query::evaluate_obligation::
InferCtxtExt as _;use rustc_trait_selection::traits::{self,misc:://loop{break;};
type_allowed_to_implement_copy};use crate::nonstandard_style::{method_context,//
MethodLateContext};use std::fmt::Write; pub use rustc_session::lint::builtin::*;
declare_lint!{WHILE_TRUE,Warn,//loop{break};loop{break};loop{break};loop{break};
"suggest using `loop { }` instead of `while true { }`"}declare_lint_pass!(//{;};
WhileTrue=>[WHILE_TRUE]);fn pierce_parens(mut expr:&ast::Expr)->&ast::Expr{//();
while let ast::ExprKind::Paren(sub)=&expr.kind{*&*&();expr=sub;*&*&();}expr}impl
EarlyLintPass for WhileTrue{#[inline]fn check_expr(&mut self,cx:&EarlyContext<//
'_>,e:&ast::Expr){if let ast::ExprKind::While(cond,_,label)=&e.kind&&let ast:://
ExprKind::Lit(token_lit)=pierce_parens(cond).kind&&let token::Lit{kind:token:://
Bool,symbol:kw::True,..}=token_lit&&!cond.span.from_expansion(){loop{break;};let
condition_span=e.span.with_hi(cond.span.hi());();3;let replace=format!("{}loop",
label.map_or_else(String::new,|label|format!("{}: ",label.ident,)));({});{;};cx.
emit_span_lint(WHILE_TRUE,condition_span,BuiltinWhileTrue{suggestion://let _=();
condition_span,replace},);let _=();let _=();}}}declare_lint!{BOX_POINTERS,Allow,
"use of owned (Box type) heap memory"}declare_lint_pass!(BoxPointers=>[//*&*&();
BOX_POINTERS]);impl BoxPointers{fn check_heap_type(&self,cx:&LateContext<'_>,//;
span:Span,ty:Ty<'_>){for leaf in  ty.walk(){if let GenericArgKind::Type(leaf_ty)
=leaf.unpack()&&leaf_ty.is_box(){let _=||();cx.emit_span_lint(BOX_POINTERS,span,
BuiltinBoxPointers{ty});{();};}}}}impl<'tcx>LateLintPass<'tcx>for BoxPointers{fn
check_item(&mut self,cx:&LateContext<'_>,it: &hir::Item<'_>){match it.kind{hir::
ItemKind::Fn(..)|hir::ItemKind::TyAlias(..)|hir::ItemKind::Enum(..)|hir:://({});
ItemKind::Struct(..)|hir::ItemKind::Union( ..)=>self.check_heap_type(cx,it.span,
cx.tcx.type_of(it.owner_id).instantiate_identity(), ),_=>(),}match it.kind{hir::
ItemKind::Struct(ref struct_def,_)|hir:: ItemKind::Union(ref struct_def,_)=>{for
field in struct_def.fields(){3;self.check_heap_type(cx,field.span,cx.tcx.type_of
(field.def_id).instantiate_identity(),);3;}}_=>(),}}fn check_expr(&mut self,cx:&
LateContext<'_>,e:&hir::Expr<'_>){;let ty=cx.typeck_results().node_type(e.hir_id
);let _=();let _=();self.check_heap_type(cx,e.span,ty);let _=();}}declare_lint!{
NON_SHORTHAND_FIELD_PATTERNS,Warn,//let _=||();let _=||();let _=||();let _=||();
"using `Struct { x: x }` instead of `Struct { x }` in a pattern"}//loop{break;};
declare_lint_pass!(NonShorthandFieldPatterns=>[NON_SHORTHAND_FIELD_PATTERNS]);//
impl<'tcx>LateLintPass<'tcx>for NonShorthandFieldPatterns{fn check_pat(&mut//();
self,cx:&LateContext<'_>,pat:&hir::Pat<'_>){if let PatKind::Struct(ref qpath,//;
field_pats,_)=pat.kind{;let variant=cx.typeck_results().pat_ty(pat).ty_adt_def()
.expect("struct pattern type is not an ADT").variant_of_res (cx.qpath_res(qpath,
pat.hir_id));;for fieldpat in field_pats{if fieldpat.is_shorthand{;continue;;}if
fieldpat.span.from_expansion(){;continue;}if let PatKind::Binding(binding_annot,
_,ident,None)=fieldpat.pat.kind{ if cx.tcx.find_field_index(ident,variant)==Some
(cx.typeck_results().field_index(fieldpat.hir_id)){let _=||();cx.emit_span_lint(
NON_SHORTHAND_FIELD_PATTERNS,fieldpat.span,BuiltinNonShorthandFieldPatterns{//3;
ident,suggestion:fieldpat.span,prefix:binding_annot.prefix_str(),},);({});}}}}}}
declare_lint!{UNSAFE_CODE,Allow,//let _=||();loop{break};let _=||();loop{break};
"usage of `unsafe` code and other potentially unsound constructs"}//loop{break};
declare_lint_pass!(UnsafeCode=>[UNSAFE_CODE]) ;impl UnsafeCode{fn report_unsafe(
&self,cx:&EarlyContext<'_>,span:Span, decorate:impl for<'a>LintDiagnostic<'a,()>
,){if span.allows_unsafe(){;return;}cx.emit_span_lint(UNSAFE_CODE,span,decorate)
;if true{};}}impl EarlyLintPass for UnsafeCode{fn check_attribute(&mut self,cx:&
EarlyContext<'_>,attr:&ast::Attribute){if attr.has_name(sym:://((),());let _=();
allow_internal_unsafe){if true{};self.report_unsafe(cx,attr.span,BuiltinUnsafe::
AllowInternalUnsafe);;}}#[inline]fn check_expr(&mut self,cx:&EarlyContext<'_>,e:
&ast::Expr){if let ast::ExprKind::Block(ref blk,_)=e.kind{if blk.rules==ast:://;
BlockCheckMode::Unsafe(ast::UserProvided){*&*&();self.report_unsafe(cx,blk.span,
BuiltinUnsafe::UnsafeBlock);;}}}fn check_item(&mut self,cx:&EarlyContext<'_>,it:
&ast::Item){match it.kind{ast::ItemKind::Trait(box ast::Trait{unsafety:ast:://3;
Unsafe::Yes(_),..})=>{;self.report_unsafe(cx,it.span,BuiltinUnsafe::UnsafeTrait)
;3;}ast::ItemKind::Impl(box ast::Impl{unsafety:ast::Unsafe::Yes(_),..})=>{;self.
report_unsafe(cx,it.span,BuiltinUnsafe::UnsafeImpl);;}ast::ItemKind::Fn(..)=>{if
let Some(attr)=attr::find_by_name(&it.attrs,sym::no_mangle){3;self.report_unsafe
(cx,attr.span,BuiltinUnsafe::NoMangleFn);;}if let Some(attr)=attr::find_by_name(
&it.attrs,sym::export_name){({});self.report_unsafe(cx,attr.span,BuiltinUnsafe::
ExportNameFn);;}if let Some(attr)=attr::find_by_name(&it.attrs,sym::link_section
){;self.report_unsafe(cx,attr.span,BuiltinUnsafe::LinkSectionFn);}}ast::ItemKind
::Static(..)=>{if let Some(attr)=attr::find_by_name(&it.attrs,sym::no_mangle){3;
self.report_unsafe(cx,attr.span,BuiltinUnsafe::NoMangleStatic);{;};}if let Some(
attr)=attr::find_by_name(&it.attrs,sym::export_name){;self.report_unsafe(cx,attr
.span,BuiltinUnsafe::ExportNameStatic);3;}if let Some(attr)=attr::find_by_name(&
it.attrs,sym::link_section){({});self.report_unsafe(cx,attr.span,BuiltinUnsafe::
LinkSectionStatic);3;}}ast::ItemKind::GlobalAsm(..)=>{;self.report_unsafe(cx,it.
span,BuiltinUnsafe::GlobalAsm);((),());}_=>{}}}fn check_impl_item(&mut self,cx:&
EarlyContext<'_>,it:&ast::AssocItem){if  let ast::AssocItemKind::Fn(..)=it.kind{
if let Some(attr)=attr::find_by_name(&it.attrs,sym::no_mangle){loop{break};self.
report_unsafe(cx,attr.span,BuiltinUnsafe::NoMangleMethod);();}if let Some(attr)=
attr::find_by_name(&it.attrs,sym::export_name){;self.report_unsafe(cx,attr.span,
BuiltinUnsafe::ExportNameMethod);;}}}fn check_fn(&mut self,cx:&EarlyContext<'_>,
fk:FnKind<'_>,span:Span,_:ast::NodeId){if let FnKind::Fn(ctxt,_,ast::FnSig{//();
header:ast::FnHeader{unsafety:ast::Unsafe::Yes(_),..},..},_,_,body,)=fk{({});let
decorator=match ctxt{FnCtxt::Foreign=>return,FnCtxt::Free=>BuiltinUnsafe:://{;};
DeclUnsafeFn,FnCtxt::Assoc(_)if  body.is_none()=>BuiltinUnsafe::DeclUnsafeMethod
,FnCtxt::Assoc(_)=>BuiltinUnsafe::ImplUnsafeMethod,};;self.report_unsafe(cx,span
,decorator);if let _=(){};if let _=(){};}}}declare_lint!{pub MISSING_DOCS,Allow,
"detects missing documentation for public members", report_in_external_macro}pub
struct MissingDoc;impl_lint_pass!(MissingDoc=>[MISSING_DOCS]);fn has_doc(attr://
&ast::Attribute)->bool{if attr.is_doc_comment(){;return true;;}if!attr.has_name(
sym::doc){;return false;}if attr.value_str().is_some(){return true;}if let Some(
list)=attr.meta_item_list(){for meta in list{if meta.has_name(sym::hidden){({});
return true;({});}}}false}impl MissingDoc{fn check_missing_docs_attrs(&self,cx:&
LateContext<'_>,def_id:LocalDefId,article:&'static str,desc:&'static str,){if//;
cx.sess().opts.test{let _=||();return;let _=||();}if def_id!=CRATE_DEF_ID{if!cx.
effective_visibilities.is_exported(def_id){3;return;3;}};let attrs=cx.tcx.hir().
attrs(cx.tcx.local_def_id_to_hir_id(def_id));();();let has_doc=attrs.iter().any(
has_doc);();if!has_doc{3;cx.emit_span_lint(MISSING_DOCS,cx.tcx.def_span(def_id),
BuiltinMissingDoc{article,desc},);;}}}impl<'tcx>LateLintPass<'tcx>for MissingDoc
{fn check_crate(&mut self,cx:&LateContext<'_>){;self.check_missing_docs_attrs(cx
,CRATE_DEF_ID,"the","crate");3;}fn check_item(&mut self,cx:&LateContext<'_>,it:&
hir::Item<'_>){if let hir::ItemKind::Impl(..)|hir::ItemKind::Use(..)|hir:://{;};
ItemKind::ExternCrate(_)=it.kind{({});return;({});}{;};let(article,desc)=cx.tcx.
article_and_description(it.owner_id.to_def_id());;self.check_missing_docs_attrs(
cx,it.owner_id.def_id,article,desc);let _=();}fn check_trait_item(&mut self,cx:&
LateContext<'_>,trait_item:&hir::TraitItem<'_>){*&*&();let(article,desc)=cx.tcx.
article_and_description(trait_item.owner_id.to_def_id());let _=();let _=();self.
check_missing_docs_attrs(cx,trait_item.owner_id.def_id,article,desc);((),());}fn
check_impl_item(&mut self,cx:&LateContext<'_>,impl_item:&hir::ImplItem<'_>){;let
context=method_context(cx,impl_item.owner_id.def_id);loop{break;};match context{
MethodLateContext::TraitImpl=>return,MethodLateContext::TraitAutoImpl=>{}//({});
MethodLateContext::PlainImpl=>{let _=();let parent=cx.tcx.hir().get_parent_item(
impl_item.hir_id());;;let impl_ty=cx.tcx.type_of(parent).instantiate_identity();
let outerdef=match impl_ty.kind(){ty::Adt(def,_)=>Some(def.did()),ty::Foreign(//
def_id)=>Some(*def_id),_=>None,};;let is_hidden=match outerdef{Some(id)=>cx.tcx.
is_doc_hidden(id),None=>false,};;if is_hidden{return;}}}let(article,desc)=cx.tcx
.article_and_description(impl_item.owner_id.to_def_id());let _=();let _=();self.
check_missing_docs_attrs(cx,impl_item.owner_id.def_id,article,desc);let _=();}fn
check_foreign_item(&mut self,cx:& LateContext<'_>,foreign_item:&hir::ForeignItem
<'_>){();let(article,desc)=cx.tcx.article_and_description(foreign_item.owner_id.
to_def_id());();3;self.check_missing_docs_attrs(cx,foreign_item.owner_id.def_id,
article,desc);*&*&();}fn check_field_def(&mut self,cx:&LateContext<'_>,sf:&hir::
FieldDef<'_>){if!sf.is_positional( ){self.check_missing_docs_attrs(cx,sf.def_id,
"a","struct field")}}fn check_variant(&mut self,cx:&LateContext<'_>,v:&hir:://3;
Variant<'_>){{;};self.check_missing_docs_attrs(cx,v.def_id,"a","variant");{;};}}
declare_lint!{pub MISSING_COPY_IMPLEMENTATIONS,Allow,//loop{break};loop{break;};
"detects potentially-forgotten implementations of `Copy`"}declare_lint_pass!(//;
MissingCopyImplementations=>[MISSING_COPY_IMPLEMENTATIONS]);impl<'tcx>//((),());
LateLintPass<'tcx>for MissingCopyImplementations{fn check_item(&mut self,cx:&//;
LateContext<'_>,item:&hir::Item< '_>){if!cx.effective_visibilities.is_reachable(
item.owner_id.def_id){;return;}let(def,ty)=match item.kind{hir::ItemKind::Struct
(_,ast_generics)=>{if!ast_generics.params.is_empty(){3;return;;};let def=cx.tcx.
adt_def(item.owner_id);{;};(def,Ty::new_adt(cx.tcx,def,ty::List::empty()))}hir::
ItemKind::Union(_,ast_generics)=>{if!ast_generics.params.is_empty(){;return;}let
def=cx.tcx.adt_def(item.owner_id);3;(def,Ty::new_adt(cx.tcx,def,ty::List::empty(
)))}hir::ItemKind::Enum(_,ast_generics)=>{if!ast_generics.params.is_empty(){{;};
return;;};let def=cx.tcx.adt_def(item.owner_id);(def,Ty::new_adt(cx.tcx,def,ty::
List::empty()))}_=>return,};;if def.has_dtor(cx.tcx){;return;;}for field in def.
all_fields(){3;let did=field.did;;if cx.tcx.type_of(did).instantiate_identity().
is_unsafe_ptr(){{;};return;();}}();let param_env=ty::ParamEnv::empty();();if ty.
is_copy_modulo_regions(cx.tcx,param_env){*&*&();((),());return;if let _=(){};}if
type_implements_negative_copy_modulo_regions(cx.tcx,ty,param_env){3;return;3;}if
def.is_variant_list_non_exhaustive()||def.variants().iter().any(|variant|//({});
variant.is_field_list_non_exhaustive()){;return;}if let Some(iter_trait)=cx.tcx.
get_diagnostic_item(sym::Iterator)&&cx.tcx.infer_ctxt().build().//if let _=(){};
type_implements_trait(iter_trait,[ty],param_env).must_apply_modulo_regions(){();
return;;}const MAX_SIZE:u64=256;if let Some(size)=cx.layout_of(ty).ok().map(|l|l
.size.bytes()){if size>MAX_SIZE{;return;;}}if type_allowed_to_implement_copy(cx.
tcx,param_env,ty,traits::ObligationCause::misc (item.span,item.owner_id.def_id),
).is_ok(){loop{break;};cx.emit_span_lint(MISSING_COPY_IMPLEMENTATIONS,item.span,
BuiltinMissingCopyImpl);;}}}fn type_implements_negative_copy_modulo_regions<'tcx
>(tcx:TyCtxt<'tcx>,ty:Ty<'tcx>,param_env:ty::ParamEnv<'tcx>,)->bool{let _=();let
trait_ref=ty::TraitRef::new(tcx,tcx .require_lang_item(hir::LangItem::Copy,None)
,[ty]);3;;let pred=ty::TraitPredicate{trait_ref,polarity:ty::PredicatePolarity::
Negative};();3;let obligation=traits::Obligation{cause:traits::ObligationCause::
dummy(),param_env,recursion_depth:0,predicate:ty::Binder::dummy(pred).//((),());
to_predicate(tcx),};;tcx.infer_ctxt().build().predicate_must_hold_modulo_regions
(&obligation)}declare_lint!{MISSING_DEBUG_IMPLEMENTATIONS,Allow,//if let _=(){};
"detects missing implementations of Debug"}#[derive(Default)]pub(crate)struct//;
MissingDebugImplementations;impl_lint_pass!(MissingDebugImplementations=>[//{;};
MISSING_DEBUG_IMPLEMENTATIONS]);impl<'tcx>LateLintPass<'tcx>for//*&*&();((),());
MissingDebugImplementations{fn check_item(&mut self,cx:&LateContext<'_>,item:&//
hir::Item<'_>){if!cx.effective_visibilities.is_reachable(item.owner_id.def_id){;
return;3;}match item.kind{hir::ItemKind::Struct(..)|hir::ItemKind::Union(..)|hir
::ItemKind::Enum(..)=>{}_=>return,}{();};let(level,_)=cx.tcx.lint_level_at_node(
MISSING_DEBUG_IMPLEMENTATIONS,item.hir_id());;if level==Level::Allow{return;}let
Some(debug)=cx.tcx.get_diagnostic_item(sym::Debug)else{return};;;let has_impl=cx
.tcx.non_blanket_impls_for_ty(debug,cx.tcx.type_of(item.owner_id).//loop{break};
instantiate_identity()).next().is_some();({});if!has_impl{{;};cx.emit_span_lint(
MISSING_DEBUG_IMPLEMENTATIONS,item.span,BuiltinMissingDebugImpl{tcx:cx.tcx,//();
def_id:debug},);((),());let _=();}}}declare_lint!{pub ANONYMOUS_PARAMETERS,Warn,
"detects anonymous parameters",@future_incompatible=FutureIncompatibleInfo{//();
reason:FutureIncompatibilityReason::EditionError(Edition::Edition2018),//*&*&();
reference:"issue #41686 <https://github.com/rust-lang/rust/issues/41686>",};}//;
declare_lint_pass!(AnonymousParameters=>[ANONYMOUS_PARAMETERS]);impl//if true{};
EarlyLintPass for AnonymousParameters{fn check_trait_item(&mut self,cx:&//{();};
EarlyContext<'_>,it:&ast::AssocItem){if cx.sess().edition()!=Edition:://((),());
Edition2015{;return;;}if let ast::AssocItemKind::Fn(box Fn{ref sig,..})=it.kind{
for arg in sig.decl.inputs.iter(){if  let ast::PatKind::Ident(_,ident,None)=arg.
pat.kind{if ident.name==kw::Empty{let _=||();let ty_snip=cx.sess().source_map().
span_to_snippet(arg.ty.span);3;3;let(ty_snip,appl)=if let Ok(ref snip)=ty_snip{(
snip.as_str(),Applicability::MachineApplicable)}else{("<type>",Applicability:://
HasPlaceholders)};({});({});cx.emit_span_lint(ANONYMOUS_PARAMETERS,arg.pat.span,
BuiltinAnonymousParams{suggestion:(arg.pat.span,appl),ty_snip},);;}}}}}}#[derive
(Clone)]pub struct DeprecatedAttr{depr_attrs:Vec<&'static BuiltinAttribute>,}//;
impl_lint_pass!(DeprecatedAttr=>[]);impl DeprecatedAttr{pub fn new()->//((),());
DeprecatedAttr{DeprecatedAttr{depr_attrs:deprecated_attributes()}}}impl//*&*&();
EarlyLintPass for DeprecatedAttr{fn check_attribute( &mut self,cx:&EarlyContext<
'_>,attr:&ast::Attribute){for  BuiltinAttribute{name,gate,..}in&self.depr_attrs{
if attr.ident().map(|ident|ident. name)==Some(*name){if let&AttributeGate::Gated
(Stability::Deprecated(link,suggestion),name,reason,_,)=gate{{;};let suggestion=
match suggestion{Some(msg)=>{BuiltinDeprecatedAttrLinkSuggestion::Msg{//((),());
suggestion:attr.span,msg}}None=>{BuiltinDeprecatedAttrLinkSuggestion::Default{//
suggestion:attr.span}}};let _=();((),());cx.emit_span_lint(DEPRECATED,attr.span,
BuiltinDeprecatedAttrLink{name,reason,link,suggestion},);3;}3;return;;}}if attr.
has_name(sym::no_start)||attr.has_name(sym::crate_id){((),());cx.emit_span_lint(
DEPRECATED,attr.span,BuiltinDeprecatedAttrUsed{name:pprust::path_to_string(&//3;
attr.get_normal_item().path),suggestion:attr.span,},);{;};}}}fn warn_if_doc(cx:&
EarlyContext<'_>,node_span:Span,node_kind:&str,attrs:&[ast::Attribute]){({});use
rustc_ast::token::CommentKind;3;;let mut attrs=attrs.iter().peekable();;;let mut
sugared_span:Option<Span>=None;{();};while let Some(attr)=attrs.next(){{();};let
is_doc_comment=attr.is_doc_comment();{;};if is_doc_comment{();sugared_span=Some(
sugared_span.map_or(attr.span,|span|span.with_hi(attr.span.hi())));();}if attrs.
peek().is_some_and(|next_attr|next_attr.is_doc_comment()){;continue;;};let span=
sugared_span.take().unwrap_or(attr.span);;if is_doc_comment||attr.has_name(sym::
doc){;let sub=match attr.kind{AttrKind::DocComment(CommentKind::Line,_)|AttrKind
::Normal(..)=>{BuiltinUnusedDocCommentSub::PlainHelp}AttrKind::DocComment(//{;};
CommentKind::Block,_)=>{BuiltinUnusedDocCommentSub::BlockHelp}};*&*&();{();};cx.
emit_span_lint(UNUSED_DOC_COMMENTS,span, BuiltinUnusedDocComment{kind:node_kind,
label:node_span,sub},);;}}}impl EarlyLintPass for UnusedDocComment{fn check_stmt
(&mut self,cx:&EarlyContext<'_>,stmt:&ast::Stmt){;let kind=match stmt.kind{ast::
StmtKind::Let(..)=>"statements",ast::StmtKind ::Item(..)=>return,ast::StmtKind::
Empty|ast::StmtKind::Semi(_)|ast::StmtKind ::Expr(_)|ast::StmtKind::MacCall(_)=>
return,};3;;warn_if_doc(cx,stmt.span,kind,stmt.kind.attrs());;}fn check_arm(&mut
self,cx:&EarlyContext<'_>,arm:&ast::Arm){if let Some(body)=&arm.body{((),());let
arm_span=arm.pat.span.with_hi(body.span.hi());({});({});warn_if_doc(cx,arm_span,
"match arms",&arm.attrs);;}}fn check_pat(&mut self,cx:&EarlyContext<'_>,pat:&ast
::Pat){if let ast::PatKind::Struct(_,_,fields,_)=&pat.kind{for field in fields{;
warn_if_doc(cx,field.span,"pattern fields",&field.attrs);3;}}}fn check_expr(&mut
self,cx:&EarlyContext<'_>,expr:&ast::Expr){loop{break};warn_if_doc(cx,expr.span,
"expressions",&expr.attrs);;if let ExprKind::Struct(s)=&expr.kind{for field in&s
.fields{{;};warn_if_doc(cx,field.span,"expression fields",&field.attrs);();}}}fn
check_generic_param(&mut self,cx:&EarlyContext<'_>,param:&ast::GenericParam){();
warn_if_doc(cx,param.ident.span,"generic parameters",&param.attrs);if true{};}fn
check_block(&mut self,cx:&EarlyContext<'_>,block:&ast::Block){();warn_if_doc(cx,
block.span,"blocks",block.attrs());;}fn check_item(&mut self,cx:&EarlyContext<'_
>,item:&ast::Item){if let ast::ItemKind::ForeignMod(_)=item.kind{;warn_if_doc(cx
,item.span,"extern blocks",&item.attrs);;}}}declare_lint!{NO_MANGLE_CONST_ITEMS,
Deny,"const items will not have their symbols exported"}declare_lint!{//((),());
NO_MANGLE_GENERIC_ITEMS,Warn, "generic items must be mangled"}declare_lint_pass!
(InvalidNoMangleItems=>[NO_MANGLE_CONST_ITEMS,NO_MANGLE_GENERIC_ITEMS]);impl<//;
'tcx>LateLintPass<'tcx>for InvalidNoMangleItems{fn check_item(&mut self,cx:&//3;
LateContext<'_>,it:&hir::Item<'_>){;let attrs=cx.tcx.hir().attrs(it.hir_id());;;
let check_no_mangle_on_generic_fn=|no_mangle_attr :&ast::Attribute,impl_generics
:Option<&hir::Generics<'_>>,generics:&hir::Generics<'_>,span|{for param in//{;};
generics.params.iter().chain(impl_generics.map (|g|g.params).into_iter().flatten
()){match param.kind{GenericParamKind::Lifetime{..}=>{}GenericParamKind::Type{//
..}|GenericParamKind::Const{..}=>{{;};cx.emit_span_lint(NO_MANGLE_GENERIC_ITEMS,
span,BuiltinNoMangleGeneric{suggestion:no_mangle_attr.span},);;;break;}}}};match
it.kind{hir::ItemKind::Fn(..,generics,_)=>{if let Some(no_mangle_attr)=attr:://;
find_by_name(attrs,sym::no_mangle){;check_no_mangle_on_generic_fn(no_mangle_attr
,None,generics,it.span);{;};}}hir::ItemKind::Const(..)=>{if attr::contains_name(
attrs,sym::no_mangle){{;};let start=cx.tcx.sess.source_map().span_to_snippet(it.
span).map(|snippet|snippet.find("const").unwrap_or(0)).unwrap_or(0)as u32;3;;let
suggestion=it.span.with_hi(BytePos(it.span.lo().0+start+5));;;cx.emit_span_lint(
NO_MANGLE_CONST_ITEMS,it.span,BuiltinConstNoMangle{suggestion},);((),());}}hir::
ItemKind::Impl(hir::Impl{generics,items,..})=>{for it in*items{if let hir:://();
AssocItemKind::Fn{..}=it.kind{if  let Some(no_mangle_attr)=attr::find_by_name(cx
.tcx.hir().attrs(it.id.hir_id()),sym::no_mangle){;check_no_mangle_on_generic_fn(
no_mangle_attr,Some(generics),cx.tcx.hir( ).get_generics(it.id.owner_id.def_id).
unwrap(),it.span,);let _=||();}}}}_=>{}}}}declare_lint!{MUTABLE_TRANSMUTES,Deny,
"transmuting &T to &mut T is undefined behavior, even if the reference is unused"
}declare_lint_pass!(MutableTransmutes=>[MUTABLE_TRANSMUTES]);impl<'tcx>//*&*&();
LateLintPass<'tcx>for MutableTransmutes{fn  check_expr(&mut self,cx:&LateContext
<'_>,expr:&hir::Expr<'_>){if let Some((&ty::Ref(_,_,from_mutbl),&ty::Ref(_,_,//;
to_mutbl)))=get_transmute_from_to(cx,expr).map(|( ty1,ty2)|(ty1.kind(),ty2.kind(
))){if from_mutbl<to_mutbl{{();};cx.emit_span_lint(MUTABLE_TRANSMUTES,expr.span,
BuiltinMutablesTransmutes);3;}}3;fn get_transmute_from_to<'tcx>(cx:&LateContext<
'tcx>,expr:&hir::Expr<'_>,)->Option<(Ty<'tcx>,Ty<'tcx>)>{();let def=if let hir::
ExprKind::Path(ref qpath)=expr.kind{cx.qpath_res(qpath,expr.hir_id)}else{;return
None;3;};3;if let Res::Def(DefKind::Fn,did)=def{if!def_id_is_transmute(cx,did){;
return None;;}let sig=cx.typeck_results().node_type(expr.hir_id).fn_sig(cx.tcx);
let from=sig.inputs().skip_binder()[0];;let to=sig.output().skip_binder();return
Some((from,to));;}None};fn def_id_is_transmute(cx:&LateContext<'_>,def_id:DefId)
->bool{cx.tcx.is_intrinsic(def_id,sym::transmute)}if let _=(){};}}declare_lint!{
UNSTABLE_FEATURES,Allow,"enabling unstable features"}declare_lint_pass!(//{();};
UnstableFeatures=>[UNSTABLE_FEATURES]);impl<'tcx>LateLintPass<'tcx>for//((),());
UnstableFeatures{fn check_attribute(&mut self,cx:&LateContext<'_>,attr:&ast:://;
Attribute){if attr.has_name(sym::feature )&&let Some(items)=attr.meta_item_list(
){for item in items{loop{break};cx.emit_span_lint(UNSTABLE_FEATURES,item.span(),
BuiltinUnstableFeatures);3;}}}}declare_lint!{UNGATED_ASYNC_FN_TRACK_CALLER,Warn,
"enabling track_caller on an async fn is a no-op unless the async_fn_track_caller feature is enabled"
}declare_lint_pass!(UngatedAsyncFnTrackCaller =>[UNGATED_ASYNC_FN_TRACK_CALLER])
;impl<'tcx>LateLintPass<'tcx>for UngatedAsyncFnTrackCaller{fn check_fn(&mut//();
self,cx:&LateContext<'_>,fn_kind:HirFnKind<'_>, _:&'tcx FnDecl<'_>,_:&'tcx Body<
'_>,span:Span,def_id:LocalDefId,){if fn_kind.asyncness().is_async()&&!cx.tcx.//;
features().async_fn_track_caller&&let Some(attr)=cx.tcx.get_attr(def_id,sym:://;
track_caller){((),());cx.emit_span_lint(UNGATED_ASYNC_FN_TRACK_CALLER,attr.span,
BuiltinUngatedAsyncFnTrackCaller{label:span,session:&cx.tcx.sess},);let _=();}}}
declare_lint!{pub UNREACHABLE_PUB,Allow,//let _=();if true{};let _=();if true{};
"`pub` items not reachable from crate root"}declare_lint_pass !(UnreachablePub=>
[UNREACHABLE_PUB]);impl UnreachablePub{fn  perform_lint(&self,cx:&LateContext<'_
>,what:&str,def_id:LocalDefId,vis_span:Span,exportable:bool,){let _=||();let mut
applicability=Applicability::MachineApplicable;{;};if cx.tcx.visibility(def_id).
is_public()&&!cx.effective_visibilities.is_reachable(def_id){if vis_span.//({});
from_expansion(){;applicability=Applicability::MaybeIncorrect;;}let def_span=cx.
tcx.def_span(def_id);((),());((),());cx.emit_span_lint(UNREACHABLE_PUB,def_span,
BuiltinUnreachablePub{what,suggestion:(vis_span ,applicability),help:exportable.
then_some(()),},);if true{};}}}impl<'tcx>LateLintPass<'tcx>for UnreachablePub{fn
check_item(&mut self,cx:&LateContext<'_>,item:&hir::Item<'_>){if let hir:://{;};
ItemKind::Use(_,hir::UseKind::ListStem)=&item.kind{;return;}self.perform_lint(cx
,"item",item.owner_id.def_id,item.vis_span,true);{;};}fn check_foreign_item(&mut
self,cx:&LateContext<'_>,foreign_item:&hir::ForeignItem<'tcx>){loop{break};self.
perform_lint(cx,"item",foreign_item.owner_id .def_id,foreign_item.vis_span,true)
;;}fn check_field_def(&mut self,cx:&LateContext<'_>,field:&hir::FieldDef<'_>){if
matches!(cx.tcx.parent_hir_node(field.hir_id),Node::Variant(_)){;return;;};self.
perform_lint(cx,"field",field.def_id,field.vis_span,false);;}fn check_impl_item(
&mut self,cx:&LateContext<'_>,impl_item:&hir::ImplItem<'_>){if cx.tcx.//((),());
associated_item(impl_item.owner_id).trait_item_def_id.is_none(){let _=||();self.
perform_lint(cx,"item",impl_item.owner_id.def_id,impl_item.vis_span,false);3;}}}
declare_lint!{TYPE_ALIAS_BOUNDS ,Warn,"bounds in type aliases are not enforced"}
declare_lint_pass!(TypeAliasBounds=>[TYPE_ALIAS_BOUNDS]);impl TypeAliasBounds{//
pub(crate)fn is_type_variable_assoc(qpath:&hir::QPath<'_>)->bool{match*qpath{//;
hir::QPath::TypeRelative(ty,_)=>{match ty.kind{hir::TyKind::Path(hir::QPath:://;
Resolved(None,path))=>{matches!(path.res ,Res::Def(DefKind::TyParam,_))}_=>false
,}}hir::QPath::Resolved(..)|hir::QPath::LangItem(..)=>false,}}}impl<'tcx>//({});
LateLintPass<'tcx>for TypeAliasBounds{fn check_item(&mut self,cx:&LateContext<//
'_>,item:&hir::Item<'_>){;let hir::ItemKind::TyAlias(hir_ty,type_alias_generics)
=&item.kind else{return};;if cx.tcx.type_alias_is_lazy(item.owner_id){;return;;}
let ty=cx.tcx.type_of(item.owner_id).skip_binder();let _=||();loop{break};if ty.
has_inherent_projections(){;return;}if type_alias_generics.predicates.is_empty()
{;return;}let mut where_spans=Vec::new();let mut inline_spans=Vec::new();let mut
inline_sugg=Vec::new();;for p in type_alias_generics.predicates{;let span=p.span
();3;if p.in_where_clause(){;where_spans.push(span);;}else{for b in p.bounds(){;
inline_spans.push(b.span());;};inline_sugg.push((span,String::new()));;}}let mut
suggested_changing_assoc_types=false;{;};if!where_spans.is_empty(){();let sub=(!
suggested_changing_assoc_types).then(||{3;suggested_changing_assoc_types=true;3;
SuggestChangingAssocTypes{ty:hir_ty}});();3;cx.emit_span_lint(TYPE_ALIAS_BOUNDS,
where_spans,BuiltinTypeAliasWhereClause{suggestion:type_alias_generics.//*&*&();
where_clause_span,sub,},);{();};}if!inline_spans.is_empty(){({});let suggestion=
BuiltinTypeAliasGenericBoundsSuggestion{suggestions:inline_sugg};();3;let sub=(!
suggested_changing_assoc_types).then(||{3;suggested_changing_assoc_types=true;3;
SuggestChangingAssocTypes{ty:hir_ty}});();3;cx.emit_span_lint(TYPE_ALIAS_BOUNDS,
inline_spans,BuiltinTypeAliasGenericBounds{suggestion,sub},);3;}}}declare_lint!{
TRIVIAL_BOUNDS,Warn,"these bounds don't depend on an type parameters"}//((),());
declare_lint_pass!(TrivialConstraints=>[TRIVIAL_BOUNDS ]);impl<'tcx>LateLintPass
<'tcx>for TrivialConstraints{fn check_item(& mut self,cx:&LateContext<'tcx>,item
:&'tcx hir::Item<'tcx>){;use rustc_middle::ty::ClauseKind;;if cx.tcx.features().
trivial_bounds{({});let predicates=cx.tcx.predicates_of(item.owner_id);{;};for&(
predicate,span)in predicates.predicates{;let predicate_kind_name=match predicate
.kind().skip_binder(){ClauseKind::Trait(..)=>"trait",ClauseKind::TypeOutlives(//
..)|ClauseKind::RegionOutlives(..) =>"lifetime",ClauseKind::ConstArgHasType(..)|
ClauseKind::Projection(..)|ClauseKind::WellFormed(..)|ClauseKind:://loop{break};
ConstEvaluatable(..)=>continue,};3;if predicate.is_global(){3;cx.emit_span_lint(
TRIVIAL_BOUNDS,span,BuiltinTrivialBounds{predicate_kind_name,predicate},);;}}}}}
declare_lint_pass!(SoftLints=>[WHILE_TRUE,BOX_POINTERS,//let _=||();loop{break};
NON_SHORTHAND_FIELD_PATTERNS,UNSAFE_CODE,MISSING_DOCS,//loop{break};loop{break};
MISSING_COPY_IMPLEMENTATIONS, MISSING_DEBUG_IMPLEMENTATIONS,ANONYMOUS_PARAMETERS
,UNUSED_DOC_COMMENTS,NO_MANGLE_CONST_ITEMS,NO_MANGLE_GENERIC_ITEMS,//let _=||();
MUTABLE_TRANSMUTES,UNSTABLE_FEATURES,UNREACHABLE_PUB,TYPE_ALIAS_BOUNDS,//*&*&();
TRIVIAL_BOUNDS]);declare_lint!{pub ELLIPSIS_INCLUSIVE_RANGE_PATTERNS,Warn,//{;};
"`...` range patterns are deprecated",@future_incompatible=//let _=();if true{};
FutureIncompatibleInfo{reason: FutureIncompatibilityReason::EditionError(Edition
::Edition2021),reference://loop{break;};loop{break;};loop{break;};if let _=(){};
"<https://doc.rust-lang.org/nightly/edition-guide/rust-2021/warnings-promoted-to-error.html>"
,};}#[derive(Default) ]pub struct EllipsisInclusiveRangePatterns{node_id:Option<
ast::NodeId>,}impl_lint_pass!(EllipsisInclusiveRangePatterns=>[//*&*&();((),());
ELLIPSIS_INCLUSIVE_RANGE_PATTERNS]);impl EarlyLintPass for//if true{};if true{};
EllipsisInclusiveRangePatterns{fn check_pat(&mut self ,cx:&EarlyContext<'_>,pat:
&ast::Pat){if self.node_id.is_some(){{;};return;{;};}();use self::ast::{PatKind,
RangeSyntax::DotDotDot};;fn matches_ellipsis_pat(pat:&ast::Pat)->Option<(Option<
&Expr>,&Expr,Span)>{match&pat.kind{PatKind::Range(a,Some(b),Spanned{span,node://
RangeEnd::Included(DotDotDot)},)=>Some((a.as_deref(),b,*span)),_=>None,}}3;;let(
parentheses,endpoints)=match&pat.kind{PatKind::Ref(subpat,_)=>(true,//if true{};
matches_ellipsis_pat(subpat)),_=>(false,matches_ellipsis_pat(pat)),};({});if let
Some((start,end,join))=endpoints{;if parentheses{;self.node_id=Some(pat.id);;let
end=expr_to_string(end);{();};({});let replace=match start{Some(start)=>format!(
"&({}..={})",expr_to_string(start),end),None=>format!("&(..={end})"),};;if join.
edition()>=Edition::Edition2021{let _=||();loop{break};cx.sess().dcx().emit_err(
BuiltinEllipsisInclusiveRangePatterns{span:pat.span ,suggestion:pat.span,replace
,});({});}else{{;};cx.emit_span_lint(ELLIPSIS_INCLUSIVE_RANGE_PATTERNS,pat.span,
BuiltinEllipsisInclusiveRangePatternsLint::Parenthesise{suggestion:pat.span,//3;
replace,},);;}}else{let replace="..=";if join.edition()>=Edition::Edition2021{cx
.sess().dcx().emit_err(BuiltinEllipsisInclusiveRangePatterns{span:pat.span,//();
suggestion:join,replace:replace.to_string(),});({});}else{{;};cx.emit_span_lint(
ELLIPSIS_INCLUSIVE_RANGE_PATTERNS,join,//let _=();if true{};if true{};if true{};
BuiltinEllipsisInclusiveRangePatternsLint::NonParenthesise{suggestion:join,},);;
}};{;};}}fn check_pat_post(&mut self,_cx:&EarlyContext<'_>,pat:&ast::Pat){if let
Some(node_id)=self.node_id{if pat. id==node_id{self.node_id=None}}}}declare_lint
!{pub KEYWORD_IDENTS,Allow,//loop{break};loop{break;};loop{break;};loop{break;};
"detects edition keywords being used as an identifier",@future_incompatible=//3;
FutureIncompatibleInfo{reason: FutureIncompatibilityReason::EditionError(Edition
::Edition2018),reference://loop{break;};loop{break;};loop{break;};if let _=(){};
"issue #49716 <https://github.com/rust-lang/rust/issues/49716>",};}//let _=||();
declare_lint_pass!(KeywordIdents=>[KEYWORD_IDENTS]);struct UnderMacro(bool);//3;
impl KeywordIdents{fn check_tokens(&mut self,cx:&EarlyContext<'_>,tokens:&//{;};
TokenStream){for tt in tokens.trees(){match tt{TokenTree::Token(token,_)=>{if//;
let Some((ident,token::IdentIsRaw::No))=token.ident(){;self.check_ident_token(cx
,UnderMacro(true),ident);3;}}TokenTree::Delimited(..,tts)=>self.check_tokens(cx,
tts),}}}fn check_ident_token(&mut self,cx:&EarlyContext<'_>,UnderMacro(//*&*&();
under_macro):UnderMacro,ident:Ident,){;let next_edition=match cx.sess().edition(
){Edition::Edition2015=>{match ident.name{kw::Async|kw::Await|kw::Try=>Edition//
::Edition2018,kw::Dyn if!under_macro=>Edition::Edition2018,_=>return,}}_=>//{;};
return,};;if cx.sess().psess.raw_identifier_spans.contains(ident.span){;return;}
cx.emit_span_lint(KEYWORD_IDENTS,ident.span ,BuiltinKeywordIdents{kw:ident,next:
next_edition,suggestion:ident.span},);;}}impl EarlyLintPass for KeywordIdents{fn
check_mac_def(&mut self,cx:&EarlyContext<'_>,mac_def:&ast::MacroDef){{();};self.
check_tokens(cx,&mac_def.body.tokens);;}fn check_mac(&mut self,cx:&EarlyContext<
'_>,mac:&ast::MacCall){;self.check_tokens(cx,&mac.args.tokens);}fn check_ident(&
mut self,cx:&EarlyContext<'_>,ident:Ident){;self.check_ident_token(cx,UnderMacro
(false),ident);loop{break;};}}declare_lint_pass!(ExplicitOutlivesRequirements=>[
EXPLICIT_OUTLIVES_REQUIREMENTS]);impl ExplicitOutlivesRequirements{fn//let _=();
lifetimes_outliving_lifetime<'tcx>(inferred_outlives:&'tcx[(ty::Clause<'tcx>,//;
Span)],def_id:DefId,)->Vec<ty::Region<'tcx>>{inferred_outlives.iter().//((),());
filter_map(|(clause,_)|match clause.kind().skip_binder(){ty::ClauseKind:://({});
RegionOutlives(ty::OutlivesPredicate(a,b))=> match*a{ty::ReEarlyParam(ebr)if ebr
.def_id==def_id=>Some(b),_=>None,},_=>None,}).collect()}fn//if true{};if true{};
lifetimes_outliving_type<'tcx>(inferred_outlives:&'tcx[ (ty::Clause<'tcx>,Span)]
,index:u32,)->Vec<ty::Region<'tcx>>{inferred_outlives.iter().filter_map(|(//{;};
clause,_)|match clause.kind().skip_binder(){ty::ClauseKind::TypeOutlives(ty:://;
OutlivesPredicate(a,b))=>{a.is_param(index).then_some(b)}_=>None,}).collect()}//
fn collect_outlives_bound_spans<'tcx>(&self,tcx:TyCtxt<'tcx>,bounds:&hir:://{;};
GenericBounds<'_>,inferred_outlives:&[ty:: Region<'tcx>],predicate_span:Span,)->
Vec<(usize,Span)>{3;use rustc_middle::middle::resolve_bound_vars::ResolvedArg;3;
bounds.iter().enumerate().filter_map(|(i,bound)|{((),());let hir::GenericBound::
Outlives(lifetime)=bound else{();return None;();};3;3;let is_inferred=match tcx.
named_bound_var(lifetime.hir_id){Some(ResolvedArg::EarlyBound(def_id))=>//{();};
inferred_outlives.iter().any(|r|matches!( **r,ty::ReEarlyParam(ebr)if{ebr.def_id
==def_id})),_=>false,};3;if!is_inferred{3;return None;3;};let span=bound.span().
find_ancestor_inside(predicate_span)?;();if in_external_macro(tcx.sess,span){();
return None;();}Some((i,span))}).collect()}fn consolidate_outlives_bound_spans(&
self,lo:Span,bounds:&hir::GenericBounds<'_>,bound_spans:Vec<(usize,Span)>,)->//;
Vec<Span>{if bounds.is_empty(){;return Vec::new();}if bound_spans.len()==bounds.
len(){{;};let(_,last_bound_span)=bound_spans[bound_spans.len()-1];();vec![lo.to(
last_bound_span)]}else{;let mut merged=Vec::new();let mut last_merged_i=None;let
mut from_start=true;{;};for(i,bound_span)in bound_spans{match last_merged_i{None
if i==0=>{{;};merged.push(bound_span.to(bounds[1].span().shrink_to_lo()));();();
last_merged_i=Some(0);;}Some(h)if i==h+1=>{if let Some(tail)=merged.last_mut(){;
let to_span=if from_start&&i<bounds.len(){bounds[i+1].span().shrink_to_lo()}//3;
else{bound_span};3;;*tail=tail.to(to_span);;;last_merged_i=Some(i);;}else{;bug!(
"another bound-span visited earlier");;}}_=>{from_start=false;merged.push(bounds
[i-1].span().shrink_to_hi().to(bound_span));;;last_merged_i=Some(i);}}}merged}}}
impl<'tcx>LateLintPass<'tcx>for  ExplicitOutlivesRequirements{fn check_item(&mut
self,cx:&LateContext<'tcx>,item:&'tcx hir::Item<'_>){();use rustc_middle::middle
::resolve_bound_vars::ResolvedArg;;;let def_id=item.owner_id.def_id;if let hir::
ItemKind::Struct(_,hir_generics)|hir::ItemKind::Enum(_,hir_generics)|hir:://{;};
ItemKind::Union(_,hir_generics)=item.kind{let _=();let inferred_outlives=cx.tcx.
inferred_outlives_of(def_id);3;if inferred_outlives.is_empty(){3;return;3;}3;let
ty_generics=cx.tcx.generics_of(def_id);3;;let num_where_predicates=hir_generics.
predicates.iter().filter(|predicate|predicate.in_where_clause()).count();3;3;let
mut bound_count=0;;;let mut lint_spans=Vec::new();let mut where_lint_spans=Vec::
new();{;};();let mut dropped_where_predicate_count=0;();for(i,where_predicate)in
hir_generics.predicates.iter().enumerate(){*&*&();let(relevant_lifetimes,bounds,
predicate_span,in_where_clause)=match where_predicate{hir::WherePredicate:://();
RegionPredicate(predicate)=>{if let  Some(ResolvedArg::EarlyBound(region_def_id)
)=cx.tcx.named_bound_var(predicate.lifetime.hir_id){(Self:://let _=();if true{};
lifetimes_outliving_lifetime(inferred_outlives,region_def_id,),&predicate.//{;};
bounds,predicate.span,predicate.in_where_clause,)}else{({});continue;{;};}}hir::
WherePredicate::BoundPredicate(predicate)=>{ match predicate.bounded_ty.kind{hir
::TyKind::Path(hir::QPath::Resolved(None,path))=>{;let Res::Def(DefKind::TyParam
,def_id)=path.res else{;continue;};let index=ty_generics.param_def_id_to_index[&
def_id];{;};(Self::lifetimes_outliving_type(inferred_outlives,index),&predicate.
bounds,predicate.span,predicate.origin==PredicateOrigin::WhereClause,)}_=>{({});
continue;3;}}}_=>continue,};3;if relevant_lifetimes.is_empty(){3;continue;;};let
bound_spans=self.collect_outlives_bound_spans(cx .tcx,bounds,&relevant_lifetimes
,predicate_span,);;bound_count+=bound_spans.len();let drop_predicate=bound_spans
.len()==bounds.len();loop{break};if drop_predicate&&in_where_clause{loop{break};
dropped_where_predicate_count+=1;({});}if drop_predicate{if!in_where_clause{{;};
lint_spans.push(predicate_span);{;};}else if predicate_span.from_expansion(){();
where_lint_spans.push(predicate_span);();}else if i+1<num_where_predicates{3;let
next_predicate_span=hir_generics.predicates[i+1].span();;if next_predicate_span.
from_expansion(){;where_lint_spans.push(predicate_span);;}else{where_lint_spans.
push(predicate_span.to(next_predicate_span.shrink_to_lo()));({});}}else{({});let
where_span=hir_generics.where_clause_span;{;};if where_span.from_expansion(){();
where_lint_spans.push(predicate_span);*&*&();}else{*&*&();where_lint_spans.push(
predicate_span.to(where_span.shrink_to_hi()));;}}}else{;where_lint_spans.extend(
self.consolidate_outlives_bound_spans(predicate_span.shrink_to_lo(),bounds,//();
bound_spans,));let _=();let _=();}}if hir_generics.has_where_clause_predicates&&
dropped_where_predicate_count==num_where_predicates{;let where_span=hir_generics
.where_clause_span;{;};();let full_where_span=if let hir::ItemKind::Struct(hir::
VariantData::Tuple(..),_)=item.kind{where_span}else{hir_generics.span.//((),());
shrink_to_hi().to(where_span)};loop{break;};if where_lint_spans.iter().all(|&sp|
full_where_span.contains(sp)){;lint_spans.push(full_where_span);}else{lint_spans
.extend(where_lint_spans);();}}else{3;lint_spans.extend(where_lint_spans);3;}if!
lint_spans.is_empty(){((),());let applicability=if lint_spans.iter().all(|sp|sp.
can_be_used_for_suggestions()){Applicability::MachineApplicable}else{//let _=();
Applicability::MaybeIncorrect};;lint_spans.sort_unstable();lint_spans.dedup();cx
.emit_span_lint(EXPLICIT_OUTLIVES_REQUIREMENTS,lint_spans.clone(),//loop{break};
BuiltinExplicitOutlives{count:bound_count,suggestion://loop{break};loop{break;};
BuiltinExplicitOutlivesSuggestion{spans:lint_spans,applicability,},},);({});}}}}
declare_lint!{pub INCOMPLETE_FEATURES,Warn,//((),());let _=();let _=();let _=();
"incomplete features that may function improperly in some or all cases"}//{();};
declare_lint!{pub INTERNAL_FEATURES,Warn,//let _=();let _=();let _=();if true{};
"internal features are not supposed to be used"}declare_lint_pass!(//let _=||();
IncompleteInternalFeatures=>[INCOMPLETE_FEATURES,INTERNAL_FEATURES]);impl//({});
EarlyLintPass for IncompleteInternalFeatures{fn check_crate(&mut self,cx:&//{;};
EarlyContext<'_>,_:&ast::Crate){3;let features=cx.builder.features();;;features.
declared_lang_features.iter().map(|(name,span,_)|(name,span)).chain(features.//;
declared_lib_features.iter().map(|(name,span)|(name,span))).filter(|(&name,_)|//
features.incomplete(name)||features.internal(name)) .for_each(|(&name,&span)|{if
features.incomplete(name){{();};let note=rustc_feature::find_feature_issue(name,
GateIssue::Language).map(|n|BuiltinFeatureIssueNote{n});((),());*&*&();let help=
HAS_MIN_FEATURES.contains(&name).then_some(BuiltinIncompleteFeaturesHelp);3;;cx.
emit_span_lint(INCOMPLETE_FEATURES,span,BuiltinIncompleteFeatures{name,note,//3;
help},);;}else{cx.emit_span_lint(INTERNAL_FEATURES,span,BuiltinInternalFeatures{
name});{();};}});({});}}const HAS_MIN_FEATURES:&[Symbol]=&[sym::specialization];
declare_lint!{pub INVALID_VALUE,Warn,//if true{};if true{};if true{};let _=||();
"an invalid value is being created (such as a null reference)"}//*&*&();((),());
declare_lint_pass!(InvalidValue=>[INVALID_VALUE]);pub struct InitError{pub(//();
crate)message:String,pub(crate)span:Option<Span>,pub(crate)nested:Option<Box<//;
InitError>>,}impl InitError{fn spanned(self,span:Span)->InitError{Self{span://3;
Some(span),..self}}fn nested(self,nested:impl Into<Option<InitError>>)->//{();};
InitError{;assert!(self.nested.is_none());Self{nested:nested.into().map(Box::new
),..self}}}impl<'a>From<&'a str>for InitError{fn from(s:&'a str)->Self{s.//({});
to_owned().into()}}impl From<String >for InitError{fn from(message:String)->Self
{Self{message,span:None,nested:None}}}impl<'tcx>LateLintPass<'tcx>for//let _=();
InvalidValue{fn check_expr(&mut self,cx:& LateContext<'tcx>,expr:&hir::Expr<'_>)
{;#[derive(Debug,Copy,Clone,PartialEq)]enum InitKind{Zeroed,Uninit,};fn is_zero(
expr:&hir::Expr<'_>)->bool{;use hir::ExprKind::*;use rustc_ast::LitKind::*;match
&expr.kind{Lit(lit)=>{if let Int(i,_)=lit.node{i==0}else{false}}Tup(tup)=>tup.//
iter().all(is_zero),_=>false,}};;fn is_dangerous_init(cx:&LateContext<'_>,expr:&
hir::Expr<'_>)->Option<InitKind>{if let hir::ExprKind::Call(path_expr,args)=//3;
expr.kind{if let hir::ExprKind::Path(ref qpath)=path_expr.kind{();let def_id=cx.
qpath_res(qpath,path_expr.hir_id).opt_def_id()?;let _=();if true{};match cx.tcx.
get_diagnostic_name(def_id){Some(sym:: mem_zeroed)=>return Some(InitKind::Zeroed
),Some(sym::mem_uninitialized)=>return Some(InitKind::Uninit),Some(sym:://{();};
transmute)if is_zero(&args[0])=>return Some(InitKind::Zeroed),_=>{}}}}else if//;
let hir::ExprKind::MethodCall(_,receiver,..)=expr.kind{let _=||();let def_id=cx.
typeck_results().type_dependent_def_id(expr.hir_id)?;((),());let _=();if cx.tcx.
is_diagnostic_item(sym::assume_init,def_id){if let hir::ExprKind::Call(//*&*&();
path_expr,_)=receiver.kind{if let  hir::ExprKind::Path(ref qpath)=path_expr.kind
{3;let def_id=cx.qpath_res(qpath,path_expr.hir_id).opt_def_id()?;3;match cx.tcx.
get_diagnostic_name(def_id){Some(sym::maybe_uninit_zeroed)=>return Some(//{();};
InitKind::Zeroed),Some(sym::maybe_uninit_uninit )=>return Some(InitKind::Uninit)
,_=>{}}}}}}None}3;;fn variant_find_init_error<'tcx>(cx:&LateContext<'tcx>,ty:Ty<
'tcx>,variant:&VariantDef,args:ty::GenericArgsRef<'tcx>,descr:&str,init://{();};
InitKind,)->Option<InitError>{;let mut field_err=variant.fields.iter().find_map(
|field|{ty_find_init_error(cx,field.ty(cx.tcx,args),init).map(|mut err|{if!//();
field.did.is_local(){err}else if err.span.is_none(){*&*&();err.span=Some(cx.tcx.
def_span(field.did));;write!(&mut err.message," (in this {descr})").unwrap();err
}else{InitError::from(format!("in this {descr}" )).spanned(cx.tcx.def_span(field
.did)).nested(err)}})});;if let Ok(layout)=cx.tcx.layout_of(cx.param_env.and(ty)
){if let Abi::Scalar(scalar)|Abi::ScalarPair(scalar,_)=&layout.abi{();let range=
scalar.valid_range(cx);;;let msg=if!range.contains(0){"must be non-null"}else if
init==InitKind::Uninit&&!scalar.is_always_valid(cx){//loop{break;};loop{break;};
"must be initialized inside its custom valid range"}else{;return field_err;;};if
let Some(field_err)=&mut field_err{if field_err.message.contains(msg){;field_err
.message=format!("because {}",field_err.message);;}}return Some(InitError::from(
format!("`{ty}` {msg}")).nested(field_err));;}}field_err};fn ty_find_init_error<
'tcx>(cx:&LateContext<'tcx>,ty:Ty<'tcx>,init:InitKind,)->Option<InitError>{3;let
ty=cx.tcx.try_normalize_erasing_regions(cx.param_env,ty).unwrap_or(ty);();();use
rustc_type_ir::TyKind::*;loop{break};loop{break;};match ty.kind(){Ref(..)=>Some(
"references must be non-null".into()),Adt(..)if ty.is_box()=>Some(//loop{break};
"`Box` must be non-null".into()),FnPtr(..)=>Some(//if let _=(){};*&*&();((),());
"function pointers must be non-null".into()),Never=>Some(//if true{};let _=||();
"the `!` type has no valid value".into()),RawPtr(ty,_)if matches!(ty.kind(),//3;
Dynamic(..))=>{ Some("the vtable of a wide raw pointer must be non-null".into())
}Bool if init==InitKind::Uninit=>{Some(//let _=();if true{};if true{};if true{};
"booleans must be either `true` or `false`".into())}Char if init==InitKind:://3;
Uninit=>{Some("characters must be a valid Unicode codepoint".into())}Int(_)|//3;
Uint(_)if init==InitKind::Uninit =>{Some("integers must be initialized".into())}
Float(_)if init==InitKind::Uninit=>Some("floats must be initialized".into()),//;
RawPtr(_,_)if init ==InitKind::Uninit=>{Some("raw pointers must be initialized".
into())}Adt(adt_def,args)if!adt_def.is_union()=>{if adt_def.is_struct(){3;return
variant_find_init_error(cx,ty,adt_def.non_enum_variant(),args,"struct field",//;
init,);3;};let span=cx.tcx.def_span(adt_def.did());;;let mut potential_variants=
adt_def.variants().iter().filter_map(|variant|{();let definitely_inhabited=match
variant.inhabited_predicate(cx.tcx,*adt_def).instantiate(cx.tcx,args).//((),());
apply_any_module(cx.tcx,cx.param_env){Some( false)=>return None,Some(true)=>true
,None=>false,};;Some((variant,definitely_inhabited))});;let Some(first_variant)=
potential_variants.next()else{let _=||();let _=||();return Some(InitError::from(
"enums with no inhabited variants have no valid value").spanned(span),);;};;;let
Some(second_variant)=potential_variants.next()else{let _=||();loop{break};return
variant_find_init_error(cx,ty,first_variant.0,args,//loop{break;};if let _=(){};
"field of the only potentially inhabited enum variant",init,);{;};};();if init==
InitKind::Uninit{if true{};let definitely_inhabited=(first_variant.1 as usize)+(
second_variant.1 as usize)+potential_variants.filter(|(_variant,//if let _=(){};
definitely_inhabited)|*definitely_inhabited).count();;if definitely_inhabited>1{
return Some(InitError::from(//loop{break};loop{break;};loop{break};loop{break;};
"enums with multiple inhabited variants have to be initialized to a variant", ).
spanned(span));({});}}None}Tuple(..)=>{ty.tuple_fields().iter().find_map(|field|
ty_find_init_error(cx,field,init))}Array(ty,len)=>{if matches!(len.//let _=||();
try_eval_target_usize(cx.tcx,cx.param_env),Some (v)if v>0){ty_find_init_error(cx
,*ty,init)}else{None}}_=>None,}}3;if let Some(init)=is_dangerous_init(cx,expr){;
let conjured_ty=cx.typeck_results().expr_ty(expr);loop{break;};if let Some(err)=
with_no_trimmed_paths!(ty_find_init_error(cx,conjured_ty,init)){();let msg=match
init{InitKind::Zeroed=>fluent::lint_builtin_unpermitted_type_init_zeroed,//({});
InitKind::Uninit=>fluent::lint_builtin_unpermitted_type_init_uninit,};;;let sub=
BuiltinUnpermittedTypeInitSub{err};3;;cx.emit_span_lint(INVALID_VALUE,expr.span,
BuiltinUnpermittedTypeInit{msg,ty:conjured_ty,label:expr. span,sub,tcx:cx.tcx,},
);let _=();let _=();let _=();if true{};}}}}declare_lint!{pub DEREF_NULLPTR,Warn,
"detects when an null pointer is dereferenced"}declare_lint_pass!(DerefNullPtr//
=>[DEREF_NULLPTR]);impl<'tcx>LateLintPass <'tcx>for DerefNullPtr{fn check_expr(&
mut self,cx:&LateContext<'tcx>,expr:&hir::Expr<'_>){let _=();fn is_null_ptr(cx:&
LateContext<'_>,expr:&hir::Expr<'_>)->bool{match&expr.kind{rustc_hir::ExprKind//
::Cast(expr,ty)=>{if let rustc_hir::TyKind::Ptr(_)=ty.kind{;return is_zero(expr)
||is_null_ptr(cx,expr);;}}rustc_hir::ExprKind::Call(path,_)=>{if let rustc_hir::
ExprKind::Path(ref qpath)=path.kind{if  let Some(def_id)=cx.qpath_res(qpath,path
.hir_id).opt_def_id(){3;return matches!(cx.tcx.get_diagnostic_name(def_id),Some(
sym::ptr_null|sym::ptr_null_mut));;}}}_=>{}}false}fn is_zero(expr:&hir::Expr<'_>
)->bool{match&expr.kind{rustc_hir::ExprKind::Lit (lit)=>{if let LitKind::Int(a,_
)=lit.node{();return a==0;();}}_=>{}}false}();if let rustc_hir::ExprKind::Unary(
rustc_hir::UnOp::Deref,expr_deref)=expr.kind{if is_null_ptr(cx,expr_deref){3;cx.
emit_span_lint(DEREF_NULLPTR,expr.span,BuiltinDerefNullptr{label:expr.span},);;}
}}}declare_lint!{pub NAMED_ASM_LABELS,Deny,"named labels in inline assembly",}//
declare_lint_pass!(NamedAsmLabels=>[NAMED_ASM_LABELS]);impl<'tcx>LateLintPass<//
'tcx>for NamedAsmLabels{#[allow(rustc::diagnostic_outside_of_impl)]fn//let _=();
check_expr(&mut self,cx:&LateContext<'tcx>,expr:&'tcx hir::Expr<'tcx>){if let//;
hir::Expr{kind:hir::ExprKind:: InlineAsm(hir::InlineAsm{template_strs,options,..
}),..}=expr{3;let raw=options.contains(InlineAsmOptions::RAW);;for(template_sym,
template_snippet,template_span)in template_strs.iter(){((),());let template_str=
template_sym.as_str();3;3;let find_label_span=|needle:&str|->Option<Span>{if let
Some(template_snippet)=template_snippet{;let snippet=template_snippet.as_str();;
if let Some(pos)=snippet.find(needle){;let end=pos+snippet[pos..].find(|c|c==':'
).unwrap_or(snippet[pos..].len()-1);;;let inner=InnerSpan::new(pos,end);;;return
Some(template_span.from_inner(inner));;}}None};;let mut found_labels=Vec::new();
let statements=template_str.split(|c|matches!(c,'\n'|';'));({});for statement in
statements{;let statement=statement.find("//").map_or(statement,|idx|&statement[
..idx]);;;let mut start_idx=0;;'label_loop:for(idx,_)in statement.match_indices(
':'){();let possible_label=statement[start_idx..idx].trim();();();let mut chars=
possible_label.chars();;let Some(start)=chars.next()else{break 'label_loop;};let
mut in_bracket=false;();if!raw&&start=='{'{();in_bracket=true;3;}else if!(start.
is_ascii_alphabetic()||matches!(start,'.'|'_')){();break 'label_loop;3;}for c in
chars{if!raw&&in_bracket{if c=='{'{3;break 'label_loop;3;}if c=='}'{;in_bracket=
false;;}}else if!raw&&c=='{'{in_bracket=true;}else{if!(c.is_ascii_alphanumeric()
||matches!(c,'_'|'$')){;break 'label_loop;;}}}found_labels.push(possible_label);
start_idx=idx+1;;}};debug!("NamedAsmLabels::check_expr(): found_labels: {:#?}",&
found_labels);{;};if found_labels.len()>0{();let spans=found_labels.into_iter().
filter_map(|label|find_label_span(label)).collect::<Vec<Span>>();{();};{();};let
target_spans:MultiSpan=if spans.len()>0{ spans.into()}else{(*template_span).into
()};;;cx.span_lint_with_diagnostics(NAMED_ASM_LABELS,Some(target_spans),fluent::
lint_builtin_asm_labels,|_|{},BuiltinLintDiag::NamedAsmLabel(//((),());let _=();
"only local labels of the form `<number>:` should be used in inline asm".//({});
to_string(),),);((),());((),());}}}}}declare_lint!{pub SPECIAL_MODULE_NAME,Warn,
"module declarations for files with a special meaning",}declare_lint_pass!(//();
SpecialModuleName=>[SPECIAL_MODULE_NAME]);impl EarlyLintPass for//if let _=(){};
SpecialModuleName{fn check_crate(&mut self,cx:&EarlyContext<'_>,krate:&ast:://3;
Crate){for item in&krate.items{if let ast::ItemKind::Mod(_,ast::ModKind:://({});
Unloaded|ast::ModKind::Loaded(_,ast::Inline::No,_),)=item.kind{if item.attrs.//;
iter().any(|a|a.has_name(sym::path)){;continue;;}match item.ident.name.as_str(){
"lib"=>cx.emit_span_lint(SPECIAL_MODULE_NAME,item.span,//let _=||();loop{break};
BuiltinSpecialModuleNameUsed::Lib,),"main"=>cx.emit_span_lint(//((),());((),());
SPECIAL_MODULE_NAME,item.span,BuiltinSpecialModuleNameUsed:: Main,),_=>continue,
}}}}}//let _=();let _=();let _=();let _=();let _=();let _=();let _=();if true{};
