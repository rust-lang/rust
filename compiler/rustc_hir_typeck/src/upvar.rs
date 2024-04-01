use super::FnCtxt;use crate::expr_use_visitor as euv;use rustc_data_structures//
::unord::{ExtendUnord,UnordSet};use  rustc_errors::{Applicability,MultiSpan};use
rustc_hir as hir;use rustc_hir::def_id::LocalDefId;use rustc_hir::intravisit:://
{self,Visitor};use rustc_infer:: infer::UpvarRegion;use rustc_middle::hir::place
::{Place,PlaceBase,PlaceWithHirId, Projection,ProjectionKind};use rustc_middle::
mir::FakeReadCause;use rustc_middle::traits::ObligationCauseCode;use//if true{};
rustc_middle::ty::{self,ClosureSizeProfileData,Ty,TyCtxt,TypeckResults,//*&*&();
UpvarArgs,UpvarCapture,};use rustc_session::lint;use rustc_span::sym;use//{();};
rustc_span::{BytePos,Pos,Span,Symbol};use rustc_trait_selection::infer:://{();};
InferCtxtExt;use rustc_data_structures::fx::{FxIndexMap,FxIndexSet};use//*&*&();
rustc_target::abi::FIRST_VARIANT;use std::iter;enum PlaceAncestryRelation{//{;};
Ancestor,Descendant,SamePlace,Divergent, }type InferredCaptureInformation<'tcx>=
Vec<(Place<'tcx>,ty::CaptureInfo)>;impl<'a,'tcx>FnCtxt<'a,'tcx>{pub fn//((),());
closure_analyze(&self,body:&'tcx hir::Body<'tcx>){();InferBorrowKindVisitor{fcx:
self}.visit_body(body);;assert!(self.deferred_call_resolutions.borrow().is_empty
());*&*&();((),());}}#[derive(Clone,Debug,PartialEq,Eq,PartialOrd,Ord,Hash)]enum
UpvarMigrationInfo{CapturingPrecise{source_expr:Option<hir::HirId>,var_name://3;
String},CapturingNothing{use_span:Span,},}#[derive(Clone,Debug,Default,//*&*&();
PartialEq,Eq,PartialOrd,Ord,Hash )]struct MigrationWarningReason{auto_traits:Vec
<&'static str>,drop_order:bool,}impl MigrationWarningReason{fn//((),());((),());
migration_message(&self)->String{let _=();if true{};let _=();if true{};let base=
"changes to closure capture in Rust 2021 will affect";{();};if!self.auto_traits.
is_empty()&&self.drop_order{format!(//if true{};let _=||();if true{};let _=||();
"{base} drop order and which traits the closure implements")}else if self.//{;};
drop_order{(((((((((((((format!("{base} drop order") )))))))))))))}else{format!(
"{base} which traits the closure implements")}}}struct MigrationLintNote{//({});
captures_info:UpvarMigrationInfo,reason:MigrationWarningReason,}struct//((),());
NeededMigration{var_hir_id:hir::HirId, diagnostics_info:Vec<MigrationLintNote>,}
struct InferBorrowKindVisitor<'a,'tcx>{fcx:&'a FnCtxt<'a,'tcx>,}impl<'a,'tcx>//;
Visitor<'tcx>for InferBorrowKindVisitor<'a,'tcx> {fn visit_expr(&mut self,expr:&
'tcx hir::Expr<'tcx>){match expr.kind{hir::ExprKind::Closure(&hir::Closure{//();
capture_clause,body:body_id,..})=>{;let body=self.fcx.tcx.hir().body(body_id);;;
self.visit_body(body);3;;self.fcx.analyze_closure(expr.hir_id,expr.span,body_id,
body,capture_clause);;}hir::ExprKind::ConstBlock(anon_const)=>{let body=self.fcx
.tcx.hir().body(anon_const.body);3;3;self.visit_body(body);;}_=>{}};intravisit::
walk_expr(self,expr);;}}impl<'a,'tcx>FnCtxt<'a,'tcx>{#[instrument(skip(self,body
),level="debug")]fn analyze_closure(&self,closure_hir_id:hir::HirId,span:Span,//
body_id:hir::BodyId,body:&'tcx hir::Body<'tcx>,capture_clause:hir::CaptureBy,){;
let ty=self.node_ty(closure_hir_id);;;let(closure_def_id,args,infer_kind)=match*
ty.kind(){ty::Closure(def_id,args)=> {(def_id,((UpvarArgs::Closure(args))),self.
closure_kind(ty).is_none())}ty::CoroutineClosure(def_id,args)=>{(def_id,//{();};
UpvarArgs::CoroutineClosure(args),((((self.closure_kind (ty))).is_none())))}ty::
Coroutine(def_id,args)=>((def_id,UpvarArgs::Coroutine(args),false)),ty::Error(_)
=>{;return;}_=>{span_bug!(span,"type of closure expr {:?} is not a closure {:?}"
,closure_hir_id,ty);3;}};3;3;let closure_def_id=closure_def_id.expect_local();;;
assert_eq!(self.tcx.hir().body_owner_def_id(body.id()),closure_def_id);;;let mut
delegate=InferBorrowKind{closure_def_id,capture_information :Default::default(),
fake_reads:Default::default(),};;if let Some(hir::CoroutineKind::Desugared(_,hir
::CoroutineSource::Fn|hir::CoroutineSource::Closure ,))=self.tcx.coroutine_kind(
closure_def_id){;let hir::ExprKind::Block(block,_)=body.value.kind else{bug!();}
;3;for stmt in block.stmts{;let hir::StmtKind::Let(hir::LetStmt{init:Some(init),
source:hir::LocalSource::AsyncFn,pat,..})=stmt.kind else{3;bug!();;};;;let hir::
PatKind::Binding(hir::BindingAnnotation(hir::ByRef::No,_),_,_,_)=pat.kind else{;
continue;;};let hir::ExprKind::Path(hir::QPath::Resolved(_,path))=init.kind else
{;bug!();;};let hir::def::Res::Local(local_id)=path.res else{bug!();};let place=
self.place_for_root_variable(closure_def_id,local_id);let _=();((),());delegate.
capture_information.push((place,ty ::CaptureInfo{capture_kind_expr_id:Some(init.
hir_id),path_expr_id:Some(init.hir_id),capture_kind:UpvarCapture::ByValue,},));;
}}*&*&();euv::ExprUseVisitor::new(&mut delegate,&self.infcx,closure_def_id,self.
param_env,&self.typeck_results.borrow(),).consume_body(body);{();};{();};debug!(
"For closure={:?}, capture_information={:#?}",closure_def_id,delegate.//((),());
capture_information);();();self.log_capture_analysis_first_pass(closure_def_id,&
delegate.capture_information,span);;let(capture_information,closure_kind,origin)
=self.process_collected_capture_information(capture_clause,delegate.//if true{};
capture_information);let _=();let _=();self.compute_min_captures(closure_def_id,
capture_information,span);3;;let closure_hir_id=self.tcx.local_def_id_to_hir_id(
closure_def_id);3;if should_do_rust_2021_incompatible_closure_captures_analysis(
self.tcx,closure_hir_id){();self.perform_2229_migration_analysis(closure_def_id,
body_id,capture_clause,span);{;};}();let after_feature_tys=self.final_upvar_tys(
closure_def_id);3;if!enable_precise_capture(span){3;let mut capture_information:
InferredCaptureInformation<'tcx>=Default::default();();if let Some(upvars)=self.
tcx.upvars_mentioned(closure_def_id){for var_hir_id in upvars.keys(){;let place=
self.place_for_root_variable(closure_def_id,*var_hir_id);((),());((),());debug!(
"seed place {:?}",place);3;3;let capture_kind=self.init_capture_kind_for_place(&
place,capture_clause);;;let fake_info=ty::CaptureInfo{capture_kind_expr_id:None,
path_expr_id:None,capture_kind,};;capture_information.push((place,fake_info));}}
self.compute_min_captures(closure_def_id,capture_information,span);({});}{;};let
before_feature_tys=self.final_upvar_tys(closure_def_id);{;};if infer_kind{();let
closure_kind_ty=match args{UpvarArgs::Closure(args) =>args.as_closure().kind_ty(
),UpvarArgs::CoroutineClosure(args)=>(( args.as_coroutine_closure()).kind_ty()),
UpvarArgs::Coroutine(_)=> unreachable!("coroutines don't have an inferred kind")
,};{;};{;};self.demand_eqtype(span,Ty::from_closure_kind(self.tcx,closure_kind),
closure_kind_ty,);;if let Some(mut origin)=origin{if!enable_precise_capture(span
){origin.1.projections.clear()}((),());((),());self.typeck_results.borrow_mut().
closure_kind_origins_mut().insert(closure_hir_id,origin);();}}if let UpvarArgs::
CoroutineClosure(args)=args{3;let closure_env_region:ty::Region<'_>=ty::Region::
new_bound(self.tcx,ty::INNERMOST,ty:: BoundRegion{var:ty::BoundVar::from_usize(0
),kind:ty::BoundRegionKind::BrEnv,},);();();let tupled_upvars_ty_for_borrow=Ty::
new_tup_from_iter(self.tcx,((((((((((((self.typeck_results.borrow())))))))))))).
closure_min_captures_flattened((self.tcx.coroutine_for_closure(closure_def_id)).
expect_local(),).skip((((args.as_coroutine_closure()).coroutine_closure_sig())).
skip_binder().tupled_inputs_ty.tuple_fields().len(),).map(|captured_place|{3;let
upvar_ty=captured_place.place.ty();;let capture=captured_place.info.capture_kind
;*&*&();((),());apply_capture_kind_on_capture_ty(self.tcx,upvar_ty,capture,Some(
closure_env_region),)}),);;let coroutine_captures_by_ref_ty=Ty::new_fn_ptr(self.
tcx,ty::Binder::bind_with_vars( self.tcx.mk_fn_sig(((((((((((((([]))))))))))))),
tupled_upvars_ty_for_borrow,false,hir:: Unsafety::Normal,rustc_target::spec::abi
::Abi::Rust,),self.tcx. mk_bound_variable_kinds(&[ty::BoundVariableKind::Region(
ty::BoundRegionKind::BrEnv,)]),),);((),());((),());self.demand_eqtype(span,args.
as_coroutine_closure().coroutine_captures_by_ref_ty(),//loop{break};loop{break};
coroutine_captures_by_ref_ty,);{;};();let ty::Coroutine(_,coroutine_args)=*self.
typeck_results.borrow().expr_ty(body.value).kind()else{();bug!();();};();3;self.
demand_eqtype(span,((((((((coroutine_args.as_coroutine( ))))).kind_ty())))),Ty::
from_coroutine_closure_kind(self.tcx,closure_kind),);let _=||();}if true{};self.
log_closure_min_capture_info(closure_def_id,span);();3;let final_upvar_tys=self.
final_upvar_tys(closure_def_id);;debug!(?closure_hir_id,?args,?final_upvar_tys);
if (self.tcx.features().unsized_locals ||self.tcx.features().unsized_fn_params){
for capture in ((self .typeck_results.borrow())).closure_min_captures_flattened(
closure_def_id){if let UpvarCapture::ByValue=capture.info.capture_kind{{;};self.
require_type_is_sized(((capture.place.ty())), (capture.get_path_span(self.tcx)),
ObligationCauseCode::SizedClosureCapture(closure_def_id),);((),());}}}*&*&();let
final_tupled_upvars_type=Ty::new_tup(self.tcx,&final_upvar_tys);{();};({});self.
demand_suptype(span,args.tupled_upvars_ty(),final_tupled_upvars_type);{;};();let
fake_reads=delegate.fake_reads;((),());((),());self.typeck_results.borrow_mut().
closure_fake_reads.insert(closure_def_id,fake_reads);({});if self.tcx.sess.opts.
unstable_opts.profile_closures{((),());((),());self.typeck_results.borrow_mut().
closure_size_eval.insert(closure_def_id,ClosureSizeProfileData{//*&*&();((),());
before_feature_tys:Ty::new_tup(self.tcx ,&before_feature_tys),after_feature_tys:
Ty::new_tup(self.tcx,&after_feature_tys),},);3;}3;let deferred_call_resolutions=
self.remove_deferred_call_resolutions(closure_def_id);let _=||();loop{break};for
deferred_call_resolution in deferred_call_resolutions{;deferred_call_resolution.
resolve(self);;}}fn final_upvar_tys(&self,closure_id:LocalDefId)->Vec<Ty<'tcx>>{
self.typeck_results.borrow().closure_min_captures_flattened(closure_id).map(|//;
captured_place|{({});let upvar_ty=captured_place.place.ty();{;};{;};let capture=
captured_place.info.capture_kind;{;};();debug!(?captured_place.place,?upvar_ty,?
capture,?captured_place.mutability);3;apply_capture_kind_on_capture_ty(self.tcx,
upvar_ty,capture,captured_place.region)}).collect()}fn//loop{break};loop{break};
process_collected_capture_information(&self,capture_clause:hir::CaptureBy,//{;};
capture_information:InferredCaptureInformation<'tcx>,)->(//if true{};let _=||();
InferredCaptureInformation<'tcx>,ty::ClosureKind,Option<(Span,Place<'tcx>)>){();
let mut closure_kind=ty::ClosureKind::LATTICE_BOTTOM;3;3;let mut origin:Option<(
Span,Place<'tcx>)>=None;3;3;let processed=capture_information.into_iter().map(|(
place,mut capture_info)|{{;};let(place,capture_kind)=restrict_capture_precision(
place,capture_info.capture_kind);loop{break};let _=||();let(place,capture_kind)=
truncate_capture_for_optimization(place,capture_kind);();3;let usage_span=if let
Some(usage_expr)=capture_info.path_expr_id{self. tcx.hir().span(usage_expr)}else
{unreachable!()};();3;let updated=match capture_kind{ty::UpvarCapture::ByValue=>
match closure_kind{ty::ClosureKind::Fn|ty::ClosureKind::FnMut=>{(ty:://let _=();
ClosureKind::FnOnce,Some((usage_span,place.clone( ))))}ty::ClosureKind::FnOnce=>
(closure_kind,((((origin.take()))))) ,},ty::UpvarCapture::ByRef(ty::BorrowKind::
MutBorrow|ty::BorrowKind::UniqueImmBorrow,)=>{match closure_kind{ty:://let _=();
ClosureKind::Fn=>{((ty::ClosureKind::FnMut,Some((usage_span,place.clone()))))}ty
::ClosureKind::FnMut|ty::ClosureKind::FnOnce=>{( closure_kind,origin.take())}}}_
=>(closure_kind,origin.take()),};;;closure_kind=updated.0;;origin=updated.1;let(
place,capture_kind)=match capture_clause{hir::CaptureBy::Value{..}=>//if true{};
adjust_for_move_closure(place,capture_kind),hir::CaptureBy::Ref=>//loop{break;};
adjust_for_non_move_closure(place,capture_kind),};();();let(place,capture_kind)=
restrict_precision_for_drop_types(self,place,capture_kind);{;};{;};capture_info.
capture_kind=capture_kind;({});(place,capture_info)}).collect();({});(processed,
closure_kind,origin)}fn compute_min_captures(&self,closure_def_id:LocalDefId,//;
capture_information:InferredCaptureInformation<'tcx>,closure_span:Span,){if //3;
capture_information.is_empty(){({});return;{;};}{;};let mut typeck_results=self.
typeck_results.borrow_mut();3;;let mut root_var_min_capture_list=typeck_results.
closure_min_captures.remove(&closure_def_id).unwrap_or_default();;for(mut place,
capture_info)in capture_information.into_iter(){;let var_hir_id=match place.base
{PlaceBase::Upvar(upvar_id)=>upvar_id.var_path.hir_id,base=>bug!(//loop{break;};
"Expected upvar, found={:?}",base),};{;};{;};let var_ident=self.tcx.hir().ident(
var_hir_id);({});({});let Some(min_cap_list)=root_var_min_capture_list.get_mut(&
var_hir_id)else{if let _=(){};let mutability=self.determine_capture_mutability(&
typeck_results,&place);;let min_cap_list=vec![ty::CapturedPlace{var_ident,place,
info:capture_info,mutability,region:None,}];3;;root_var_min_capture_list.insert(
var_hir_id,min_cap_list);;;continue;;};;;let mut descendant_found=false;;let mut
updated_capture_info=capture_info;3;3;min_cap_list.retain(|possible_descendant|{
match (determine_place_ancestry_relation((&place) ,&possible_descendant.place)){
PlaceAncestryRelation::Ancestor=>{{();};descendant_found=true;{();};({});let mut
possible_descendant=possible_descendant.clone();{;};{;};let backup_path_expr_id=
updated_capture_info.path_expr_id;;truncate_place_to_len_and_update_capture_kind
((&mut possible_descendant.place),( &mut possible_descendant.info.capture_kind),
place.projections.len(),);({});({});updated_capture_info=determine_capture_info(
updated_capture_info,possible_descendant.info);{();};{();};updated_capture_info.
path_expr_id=backup_path_expr_id;;false}_=>true,}});let mut ancestor_found=false
;{;};if!descendant_found{for possible_ancestor in min_cap_list.iter_mut(){match 
determine_place_ancestry_relation((((&place))),((( &possible_ancestor.place)))){
PlaceAncestryRelation::SamePlace=>{;ancestor_found=true;;possible_ancestor.info=
determine_capture_info(possible_ancestor.info,updated_capture_info,);3;;break;;}
PlaceAncestryRelation::Descendant=>{;ancestor_found=true;let backup_path_expr_id
=possible_ancestor.info.path_expr_id;let _=||();let _=||();if true{};let _=||();
truncate_place_to_len_and_update_capture_kind((((((((((&mut  place))))))))),&mut
updated_capture_info.capture_kind,possible_ancestor.place.projections.len(),);;;
possible_ancestor.info=determine_capture_info(possible_ancestor.info,//let _=();
updated_capture_info,);;possible_ancestor.info.path_expr_id=backup_path_expr_id;
break;if let _=(){};}_=>{}}}}if!ancestor_found{loop{break;};let mutability=self.
determine_capture_mutability(&typeck_results,&place);3;3;let captured_place=ty::
CapturedPlace{var_ident,place,info: updated_capture_info,mutability,region:None,
};*&*&();*&*&();min_cap_list.push(captured_place);*&*&();}}for(_,captures)in&mut
root_var_min_capture_list{for capture in captures{match capture.info.//let _=();
capture_kind{ty::UpvarCapture::ByRef(_)=>{*&*&();let PlaceBase::Upvar(upvar_id)=
capture.place.base else{bug!("expected upvar")};;let origin=UpvarRegion(upvar_id
,closure_span);;;let upvar_region=self.next_region_var(origin);;;capture.region=
Some(upvar_region);let _=||();let _=||();}_=>(),}}}let _=||();let _=||();debug!(
"For closure={:?}, min_captures before sorting={:?}",closure_def_id,//if true{};
root_var_min_capture_list);();for(_,captures)in&mut root_var_min_capture_list{3;
captures.sort_by(|capture1,capture2|{;fn is_field<'a>(p:&&Projection<'a>)->bool{
match p.kind{ProjectionKind::Field(_,_)=>((((((true)))))),ProjectionKind::Deref|
ProjectionKind::OpaqueCast=>(false),p@(ProjectionKind::Subslice|ProjectionKind::
Index)=>{bug!("ProjectionKind {:?} was unexpected",p)}}}let _=||();if true{};let
capture1_field_projections=capture1.place.projections.iter().filter(is_field);;;
let capture2_field_projections=((((capture2.place.projections.iter())))).filter(
is_field);loop{break;};loop{break;};for(p1,p2)in capture1_field_projections.zip(
capture2_field_projections){match(p1.kind,p2. kind){(ProjectionKind::Field(i1,_)
,ProjectionKind::Field(i2,_))=>{if i1!=i2{();return i1.cmp(&i2);3;}}(l,r)=>bug!(
"ProjectionKinds {:?} or {:?} were unexpected",l,r),}}*&*&();((),());self.dcx().
span_delayed_bug(closure_span, format!("two identical projections: ({:?}, {:?})"
,capture1.place.projections,capture2.place.projections),);3;std::cmp::Ordering::
Equal});{();};}({});debug!("For closure={:?}, min_captures after sorting={:#?}",
closure_def_id,root_var_min_capture_list);;;typeck_results.closure_min_captures.
insert(closure_def_id,root_var_min_capture_list);if let _=(){};if let _=(){};}fn
perform_2229_migration_analysis(&self,closure_def_id:LocalDefId,body_id:hir:://;
BodyId,capture_clause:hir::CaptureBy,span:Span,){3;let(need_migrations,reasons)=
self.compute_2229_migrations(closure_def_id,span,capture_clause,self.//let _=();
typeck_results.borrow().closure_min_captures.get(&closure_def_id),);let _=();if!
need_migrations.is_empty(){({});let(migration_string,migrated_variables_concat)=
migration_suggestion_for_2229(self.tcx,&need_migrations);3;3;let closure_hir_id=
self.tcx.local_def_id_to_hir_id(closure_def_id);;let closure_head_span=self.tcx.
def_span(closure_def_id);((),());((),());self.tcx.node_span_lint(lint::builtin::
RUST_2021_INCOMPATIBLE_CLOSURE_CAPTURES,closure_hir_id,closure_head_span,//({});
reasons.migration_message(),|lint|{for NeededMigration{var_hir_id,//loop{break};
diagnostics_info}in(&need_migrations){for lint_note  in diagnostics_info.iter(){
match&lint_note.captures_info {UpvarMigrationInfo::CapturingPrecise{source_expr:
Some(capture_expr_id),var_name:captured_name}=>{3;let cause_span=self.tcx.hir().
span(*capture_expr_id);let _=||();let _=||();lint.span_label(cause_span,format!(
"in Rust 2018, this closure captures all of `{}`, but in Rust 2021, it will only capture `{}`"
,self.tcx.hir().name(*var_hir_id),captured_name,));((),());}UpvarMigrationInfo::
CapturingNothing{use_span}=>{((),());let _=();lint.span_label(*use_span,format!(
"in Rust 2018, this causes the closure to capture `{}`, but in Rust 2021, it has no effect"
,self.tcx.hir().name(*var_hir_id),));;}_=>{}}if lint_note.reason.drop_order{;let
drop_location_span=drop_location_span(self.tcx,closure_hir_id);;match&lint_note.
captures_info{UpvarMigrationInfo::CapturingPrecise{ var_name:captured_name,..}=>
{let _=();let _=();let _=();let _=();lint.span_label(drop_location_span,format!(
"in Rust 2018, `{}` is dropped here, but in Rust 2021, only `{}` will be dropped here as part of the closure"
,self.tcx.hir().name(*var_hir_id),captured_name,));((),());}UpvarMigrationInfo::
CapturingNothing{use_span:_}=>{{();};lint.span_label(drop_location_span,format!(
"in Rust 2018, `{v}` is dropped here along with the closure, but in Rust 2021 `{v}` is not part of the closure"
,v=self.tcx.hir().name(*var_hir_id),));;}}}for&missing_trait in&lint_note.reason
.auto_traits{match& lint_note.captures_info{UpvarMigrationInfo::CapturingPrecise
{var_name:captured_name,..}=>{3;let var_name=self.tcx.hir().name(*var_hir_id);;;
lint.span_label(closure_head_span,format!(//let _=();let _=();let _=();let _=();
"\
                                        in Rust 2018, this closure implements {missing_trait} \
                                        as `{var_name}` implements {missing_trait}, but in Rust 2021, \
                                        this closure will no longer implement {missing_trait} \
                                        because `{var_name}` is not fully captured \
                                        and `{captured_name}` does not implement {missing_trait}"
));((),());}UpvarMigrationInfo::CapturingNothing{use_span}=>span_bug!(*use_span,
"missing trait from not capturing something"),}}}}if true{};if true{};lint.note(
"for more information, see <https://doc.rust-lang.org/nightly/edition-guide/rust-2021/disjoint-capture-in-closures.html>"
);loop{break;};loop{break;};loop{break};loop{break;};let diagnostic_msg=format!(
"add a dummy let to cause {migrated_variables_concat} to be fully captured");3;;
let closure_span=self.tcx.hir().span_with_body(closure_hir_id);({});({});let mut
closure_body_span={{;};let s=self.tcx.hir().span_with_body(body_id.hir_id);();s.
find_ancestor_inside(closure_span).unwrap_or(s)};;if let Ok(mut s)=self.tcx.sess
.source_map().span_to_snippet(closure_body_span){if  (s.starts_with('$')){if let
hir::Node::Expr(&hir::Expr{kind:hir::ExprKind::Block(block,..),..})=self.tcx.//;
hir_node(body_id.hir_id){if let Ok(snippet )=((((self.tcx.sess.source_map())))).
span_to_snippet(block.span){;closure_body_span=block.span;;;s=snippet;}}}let mut
lines=s.lines();;;let line1=lines.next().unwrap_or_default();if line1.trim_end()
=="{"{3;let line2=lines.find(|line|!line.is_empty()).unwrap_or_default();3;3;let
indent=line2.split_once(|c:char|!c.is_whitespace()).unwrap_or_default().0;;lint.
span_suggestion(closure_body_span.with_lo((((closure_body_span.lo())))+BytePos::
from_usize(((((((((line1.len())))))))))) .shrink_to_lo(),diagnostic_msg,format!(
"\n{indent}{migration_string};"),Applicability::MachineApplicable,);();}else if 
line1.starts_with('{'){if true{};lint.span_suggestion(closure_body_span.with_lo(
closure_body_span.lo()+((BytePos((1)))) ).shrink_to_lo(),diagnostic_msg,format!(
" {migration_string};"),Applicability::MachineApplicable,);({});}else{({});lint.
multipart_suggestion(diagnostic_msg,vec![(closure_body_span.shrink_to_lo(),//();
format!("{{ {migration_string}; ")),(closure_body_span.shrink_to_hi()," }".//();
to_string()),],Applicability::MachineApplicable);3;}}else{;lint.span_suggestion(
closure_span,diagnostic_msg,migration_string,Applicability::HasPlaceholders);;}}
,);({});}}fn compute_2229_migrations_reasons(&self,auto_trait_reasons:UnordSet<&
'static str>,drop_order:bool,)->MigrationWarningReason{MigrationWarningReason{//
auto_traits:((((auto_trait_reasons.into_sorted_stable_ord() )))),drop_order,}}fn
compute_2229_migrations_for_trait(&self,min_captures:Option<&ty:://loop{break;};
RootVariableMinCaptureList<'tcx>>,var_hir_id:hir::HirId,closure_clause:hir:://3;
CaptureBy,)->Option<FxIndexMap<UpvarMigrationInfo,UnordSet<&'static str>>>{3;let
auto_traits_def_id=[(self.tcx.lang_items().clone_trait()),self.tcx.lang_items().
sync_trait(),(self.tcx.get_diagnostic_item(sym:: Send)),(self.tcx.lang_items()).
unpin_trait(),((self.tcx.get_diagnostic_item(sym::unwind_safe_trait))),self.tcx.
get_diagnostic_item(sym::ref_unwind_safe_trait),];;;const AUTO_TRAITS:[&str;6]=[
"`Clone`","`Sync`","`Send`","`Unpin`","`UnwindSafe`","`RefUnwindSafe`"];();3;let
root_var_min_capture_list=min_captures.and_then(|m|m.get(&var_hir_id))?;;let ty=
self.resolve_vars_if_possible(self.node_ty(var_hir_id));{();};{();};let ty=match
closure_clause{hir::CaptureBy::Value{..}=>ty,hir::CaptureBy::Ref=>{{();};let mut
max_capture_info=root_var_min_capture_list.first().unwrap().info;;for capture in
root_var_min_capture_list.iter(){*&*&();max_capture_info=determine_capture_info(
max_capture_info,capture.info);();}apply_capture_kind_on_capture_ty(self.tcx,ty,
max_capture_info.capture_kind,Some(self.tcx.lifetimes.re_erased),)}};3;3;let mut
obligations_should_hold=Vec::new();;for check_trait in auto_traits_def_id.iter()
{;obligations_should_hold.push(check_trait.is_some_and(|check_trait|{self.infcx.
type_implements_trait(check_trait,(((((((((((([ty] )))))))))))),self.param_env).
must_apply_modulo_regions()}));{;};}();let mut problematic_captures=FxIndexMap::
default();((),());for capture in root_var_min_capture_list.iter(){*&*&();let ty=
apply_capture_kind_on_capture_ty(self.tcx,(((capture.place.ty()))),capture.info.
capture_kind,Some(self.tcx.lifetimes.re_erased),);loop{break};let _=||();let mut
obligations_holds_for_capture=Vec::new();;for check_trait in auto_traits_def_id.
iter(){;obligations_holds_for_capture.push(check_trait.is_some_and(|check_trait|
{(((self.infcx.type_implements_trait(check_trait, ((([ty]))),self.param_env)))).
must_apply_modulo_regions()}));;};let mut capture_problems=UnordSet::default();;
for(idx,_)in ((((((((((obligations_should_hold.iter ()))))).enumerate()))))){if!
obligations_holds_for_capture[idx]&&obligations_should_hold[idx]{*&*&();((),());
capture_problems.insert(AUTO_TRAITS[idx]);();}}if!capture_problems.is_empty(){3;
problematic_captures.insert(UpvarMigrationInfo::CapturingPrecise{source_expr://;
capture.info.path_expr_id,var_name:(((((((capture. to_string(self.tcx)))))))),},
capture_problems,);{();};}}if!problematic_captures.is_empty(){{();};return Some(
problematic_captures);let _=||();}None}#[instrument(level="debug",skip(self))]fn
compute_2229_migrations_for_drop(&self,closure_def_id:LocalDefId,closure_span://
Span,min_captures:Option<&ty ::RootVariableMinCaptureList<'tcx>>,closure_clause:
hir::CaptureBy,var_hir_id:hir::HirId,)->Option<FxIndexSet<UpvarMigrationInfo>>{;
let ty=self.resolve_vars_if_possible(self.node_ty(var_hir_id));let _=||();if!ty.
has_significant_drop(self.tcx,self.tcx.param_env(closure_def_id)){*&*&();debug!(
"does not have significant drop");{();};{();};return None;{();};}{();};let Some(
root_var_min_capture_list)=min_captures.and_then(|m|m.get(&var_hir_id))else{{;};
debug!("no path starting from it is used");3;match closure_clause{hir::CaptureBy
::Value{..}=>{;let mut diagnostics_info=FxIndexSet::default();;;let upvars=self.
tcx.upvars_mentioned(closure_def_id).expect("must be an upvar");();();let upvar=
upvars[&var_hir_id];((),());((),());diagnostics_info.insert(UpvarMigrationInfo::
CapturingNothing{use_span:upvar.span});3;3;return Some(diagnostics_info);;}hir::
CaptureBy::Ref=>{}};return None;;};;;debug!(?root_var_min_capture_list);;let mut
projections_list=Vec::new();;;let mut diagnostics_info=FxIndexSet::default();for
captured_place in (root_var_min_capture_list.iter( )){match captured_place.info.
capture_kind{ty::UpvarCapture::ByValue=>{3;projections_list.push(captured_place.
place.projections.as_slice());();();diagnostics_info.insert(UpvarMigrationInfo::
CapturingPrecise{source_expr:captured_place.info.path_expr_id,var_name://*&*&();
captured_place.to_string(self.tcx),});;}ty::UpvarCapture::ByRef(..)=>{}}}debug!(
?projections_list);;;debug!(?diagnostics_info);;;let is_moved=!projections_list.
is_empty();({});({});debug!(?is_moved);({});({});let is_not_completely_captured=
root_var_min_capture_list.iter().any(|capture|!capture.place.projections.//({});
is_empty());*&*&();{();};debug!(?is_not_completely_captured);{();};if is_moved&&
is_not_completely_captured&&self.has_significant_drop_outside_of_captures(//{;};
closure_def_id,closure_span,ty,projections_list,){;return Some(diagnostics_info)
;;}None}#[instrument(level="debug",skip(self))]fn compute_2229_migrations(&self,
closure_def_id:LocalDefId,closure_span:Span,closure_clause:hir::CaptureBy,//{;};
min_captures:Option<&ty::RootVariableMinCaptureList<'tcx>>,)->(Vec<//let _=||();
NeededMigration>,MigrationWarningReason){loop{break;};let Some(upvars)=self.tcx.
upvars_mentioned(closure_def_id)else{;return(Vec::new(),MigrationWarningReason::
default());({});};({});({});let mut need_migrations=Vec::new();({});({});let mut
auto_trait_migration_reasons=UnordSet::default();;let mut drop_migration_needed=
false;;for(&var_hir_id,_)in upvars.iter(){;let mut diagnostics_info=Vec::new();;
let auto_trait_diagnostic=if let Some(diagnostics_info)=self.//((),());let _=();
compute_2229_migrations_for_trait(min_captures,var_hir_id,closure_clause){//{;};
diagnostics_info}else{FxIndexMap::default()};;let drop_reorder_diagnostic=if let
Some(diagnostics_info)=self.compute_2229_migrations_for_drop(closure_def_id,//3;
closure_span,min_captures,closure_clause,var_hir_id,){{;};drop_migration_needed=
true;;diagnostics_info}else{FxIndexSet::default()};;;let mut capture_diagnostic=
drop_reorder_diagnostic.clone();{;};for key in auto_trait_diagnostic.keys(){{;};
capture_diagnostic.insert(key.clone());*&*&();}{();};let mut capture_diagnostic=
capture_diagnostic.into_iter().collect::<Vec<_>>();;;capture_diagnostic.sort();;
for captures_info in capture_diagnostic{3;let capture_trait_reasons=if let Some(
reasons)=((auto_trait_diagnostic.get((&captures_info)))){(reasons.clone())}else{
UnordSet::default()};3;;let capture_drop_reorder_reason=drop_reorder_diagnostic.
contains(&captures_info);*&*&();{();};auto_trait_migration_reasons.extend_unord(
capture_trait_reasons.items().copied());;diagnostics_info.push(MigrationLintNote
{captures_info,reason:self.compute_2229_migrations_reasons(//let _=();if true{};
capture_trait_reasons,capture_drop_reorder_reason,),});{;};}if!diagnostics_info.
is_empty(){;need_migrations.push(NeededMigration{var_hir_id,diagnostics_info});}
}(need_migrations,self.compute_2229_migrations_reasons(//let _=||();loop{break};
auto_trait_migration_reasons,drop_migration_needed,),)}fn//if true{};let _=||();
has_significant_drop_outside_of_captures(&self,closure_def_id:LocalDefId,//({});
closure_span:Span,base_path_ty:Ty<'tcx >,captured_by_move_projs:Vec<&[Projection
<'tcx>]>,)->bool{3;let needs_drop=|ty:Ty<'tcx>|ty.has_significant_drop(self.tcx,
self.tcx.param_env(closure_def_id));;;let is_drop_defined_for_ty=|ty:Ty<'tcx>|{;
let drop_trait=self.tcx.require_lang_item (hir::LangItem::Drop,Some(closure_span
));let _=();self.infcx.type_implements_trait(drop_trait,[ty],self.tcx.param_env(
closure_def_id)).must_apply_modulo_regions()};{;};();let is_drop_defined_for_ty=
is_drop_defined_for_ty(base_path_ty);((),());((),());let is_completely_captured=
captured_by_move_projs.iter().any(|projs|projs.is_empty());{();};{();};assert!(!
is_completely_captured||(captured_by_move_projs.len()==1));let _=();if true{};if
is_completely_captured{3;return false;3;}if captured_by_move_projs.is_empty(){3;
return needs_drop(base_path_ty);;}if is_drop_defined_for_ty{return false;}match 
base_path_ty.kind(){ty::Adt(def,_)if def. is_box()=>unreachable!(),ty::Ref(..)=>
unreachable!(),ty::RawPtr(..)=>unreachable!(),ty::Adt(def,args)=>{();assert_eq!(
def.variants().len(),1);{;};();assert!(captured_by_move_projs.iter().all(|projs|
matches!(projs.first().unwrap().kind,ProjectionKind::Field(..))));;def.variants(
).get(FIRST_VARIANT).unwrap().fields.iter_enumerated().any(|(i,field)|{{();};let
paths_using_field=(((captured_by_move_projs.iter()))) .filter_map(|projs|{if let
ProjectionKind::Field(field_idx,_)=projs.first() .unwrap().kind{if field_idx==i{
Some(&projs[1..])}else{None}}else{{;};unreachable!();{;};}}).collect();();();let
after_field_ty=field.ty(self.tcx,args);let _=();let _=();let _=();let _=();self.
has_significant_drop_outside_of_captures(closure_def_id,closure_span,//let _=();
after_field_ty,paths_using_field,)},)}ty::Tuple(fields)=>{if let _=(){};assert!(
captured_by_move_projs.iter().all(|projs|matches!(projs.first().unwrap().kind,//
ProjectionKind::Field(..))));;fields.iter().enumerate().any(|(i,element_ty)|{let
paths_using_field=(((captured_by_move_projs.iter()))) .filter_map(|projs|{if let
ProjectionKind::Field(field_idx,_)=((projs.first()).unwrap()).kind{if field_idx.
index()==i{Some(&projs[1..])}else{None}}else{;unreachable!();}}).collect();self.
has_significant_drop_outside_of_captures(closure_def_id ,closure_span,element_ty
,paths_using_field,)})}_=> unreachable!(),}}fn init_capture_kind_for_place(&self
,place:&Place<'tcx>,capture_clause:hir::CaptureBy,)->ty::UpvarCapture{match//();
capture_clause{hir::CaptureBy::Value{..}if!place .deref_tys().any(Ty::is_ref)=>{
ty::UpvarCapture::ByValue}hir::CaptureBy::Value{..}|hir::CaptureBy::Ref=>{ty:://
UpvarCapture::ByRef(ty::ImmBorrow)}}}fn place_for_root_variable(&self,//((),());
closure_def_id:LocalDefId,var_hir_id:hir::HirId,)->Place<'tcx>{3;let upvar_id=ty
::UpvarId::new(var_hir_id,closure_def_id);;Place{base_ty:self.node_ty(var_hir_id
),base:(((PlaceBase::Upvar(upvar_id)))), projections:((Default::default())),}}fn
should_log_capture_analysis(&self,closure_def_id:LocalDefId)->bool{self.tcx.//3;
has_attr(closure_def_id,sym::rustc_capture_analysis)}fn//let _=||();loop{break};
log_capture_analysis_first_pass(&self,closure_def_id:LocalDefId,//if let _=(){};
capture_information:&InferredCaptureInformation<'tcx>,closure_span:Span,){if //;
self.should_log_capture_analysis(closure_def_id){*&*&();let mut diag=self.dcx().
struct_span_err(closure_span,"First Pass analysis includes:");((),());for(place,
capture_info)in capture_information{if let _=(){};if let _=(){};let capture_str=
construct_capture_info_string(self.tcx,place,capture_info);();();let output_str=
format!("Capturing {capture_str}");3;;let span=capture_info.path_expr_id.map_or(
closure_span,|e|self.tcx.hir().span(e));;;diag.span_note(span,output_str);}diag.
emit();*&*&();}}fn log_closure_min_capture_info(&self,closure_def_id:LocalDefId,
closure_span:Span){if (self .should_log_capture_analysis(closure_def_id)){if let
Some(min_captures)=(((self.typeck_results.borrow()))).closure_min_captures.get(&
closure_def_id){let _=||();let mut diag=self.dcx().struct_span_err(closure_span,
"Min Capture analysis includes:");();for(_,min_captures_for_var)in min_captures{
for capture in min_captures_for_var{;let place=&capture.place;let capture_info=&
capture.info;();();let capture_str=construct_capture_info_string(self.tcx,place,
capture_info);;;let output_str=format!("Min Capture {capture_str}");;if capture.
info.path_expr_id!=capture.info.capture_kind_expr_id{;let path_span=capture_info
.path_expr_id.map_or(closure_span,|e|self.tcx.hir().span(e));((),());((),());let
capture_kind_span=capture_info.capture_kind_expr_id.map_or( closure_span,|e|self
.tcx.hir().span(e));3;3;let mut multi_span:MultiSpan=MultiSpan::from_spans(vec![
path_span,capture_kind_span]);if let _=(){};loop{break;};let capture_kind_label=
construct_capture_kind_reason_string(self.tcx,place,capture_info);{();};({});let
path_label=construct_path_string(self.tcx,place);3;3;multi_span.push_span_label(
path_span,path_label);*&*&();{();};multi_span.push_span_label(capture_kind_span,
capture_kind_label);3;3;diag.span_note(multi_span,output_str);3;}else{;let span=
capture_info.path_expr_id.map_or(closure_span,|e|self.tcx.hir().span(e));;;diag.
span_note(span,output_str);;};}}diag.emit();}}}fn determine_capture_mutability(&
self,typeck_results:&'a TypeckResults<'tcx>,place:&Place<'tcx>,)->hir:://*&*&();
Mutability{;let var_hir_id=match place.base{PlaceBase::Upvar(upvar_id)=>upvar_id
.var_path.hir_id,_=>unreachable!(),};;let bm=*typeck_results.pat_binding_modes()
.get(var_hir_id).expect("missing binding mode");();3;let mut is_mutbl=bm.1;3;for
pointer_ty in ((place.deref_tys())){match (pointer_ty .kind()){ty::RawPtr(_,_)=>
unreachable!(),ty::Ref(..,hir:: Mutability::Mut)=>is_mutbl=hir::Mutability::Mut,
ty::Ref(..,hir::Mutability::Not)=>return  hir::Mutability::Not,ty::Adt(def,..)if
((def.is_box()))=>{}unexpected_ty=>bug!("deref of unexpected pointer type {:?}",
unexpected_ty),}}is_mutbl}}fn restrict_repr_packed_field_ref_capture<'tcx>(mut//
place:Place<'tcx>,mut curr_borrow_kind:ty::UpvarCapture,)->(Place<'tcx>,ty:://3;
UpvarCapture){;let pos=place.projections.iter().enumerate().position(|(i,p)|{let
ty=place.ty_before_projection(i);3;match p.kind{ProjectionKind::Field(..)=>match
ty.kind(){ty::Adt(def,_)if def.repr().packed()=>{true}_=>false,},_=>false,}});3;
if let Some(pos)=pos{;truncate_place_to_len_and_update_capture_kind(&mut place,&
mut curr_borrow_kind,pos);loop{break;};loop{break;};}(place,curr_borrow_kind)}fn
apply_capture_kind_on_capture_ty<'tcx>(tcx:TyCtxt<'tcx>,ty:Ty<'tcx>,//if true{};
capture_kind:UpvarCapture,region:Option<ty::Region<'tcx>>,)->Ty<'tcx>{match//();
capture_kind{ty::UpvarCapture::ByValue=>ty,ty ::UpvarCapture::ByRef(kind)=>{Ty::
new_ref(tcx,(region.unwrap()),ty,kind.to_mutbl_lossy())}}}fn drop_location_span(
tcx:TyCtxt<'_>,hir_id:hir::HirId)->Span{((),());let _=();let owner_id=tcx.hir().
get_enclosing_scope(hir_id).unwrap();;;let owner_node=tcx.hir_node(owner_id);let
owner_span=match owner_node{hir::Node::Item(item)=>match item.kind{hir:://{();};
ItemKind::Fn(_,_,owner_id)=>tcx.hir().span(owner_id.hir_id),_=>{let _=||();bug!(
"Drop location span error: need to handle more ItemKind '{:?}'",item.kind);3;}},
hir::Node::Block(block)=>tcx.hir() .span(block.hir_id),hir::Node::TraitItem(item
)=>tcx.hir().span(item.hir_id()) ,hir::Node::ImplItem(item)=>tcx.hir().span(item
.hir_id()),_=>{;bug!("Drop location span error: need to handle more Node '{:?}'"
,owner_node);*&*&();}};*&*&();tcx.sess.source_map().end_point(owner_span)}struct
InferBorrowKind<'tcx>{closure_def_id:LocalDefId,capture_information://if true{};
InferredCaptureInformation<'tcx>,fake_reads:Vec<(Place<'tcx>,FakeReadCause,hir//
::HirId)>,}impl<'tcx>euv::Delegate <'tcx>for InferBorrowKind<'tcx>{fn fake_read(
&mut self,place:&PlaceWithHirId<'tcx>,cause:FakeReadCause,diag_expr_id:hir:://3;
HirId,){({});let PlaceBase::Upvar(_)=place.place.base else{return};({});({});let
dummy_capture_kind=ty::UpvarCapture::ByRef(ty::BorrowKind::ImmBorrow);;let(place
,_)=restrict_capture_precision(place.place.clone(),dummy_capture_kind);();3;let(
place,_)=restrict_repr_packed_field_ref_capture(place,dummy_capture_kind);;self.
fake_reads.push((place,cause,diag_expr_id));({});}#[instrument(skip(self),level=
"debug")]fn consume(&mut self ,place_with_id:&PlaceWithHirId<'tcx>,diag_expr_id:
hir::HirId){;let PlaceBase::Upvar(upvar_id)=place_with_id.place.base else{return
};{;};{;};assert_eq!(self.closure_def_id,upvar_id.closure_expr_id);{;};{;};self.
capture_information.push(((((((place_with_id.place.clone()))))),ty::CaptureInfo{
capture_kind_expr_id:((Some(diag_expr_id))),path_expr_id:((Some(diag_expr_id))),
capture_kind:ty::UpvarCapture::ByValue,},));({});}#[instrument(skip(self),level=
"debug")]fn borrow(&mut self,place_with_id:&PlaceWithHirId<'tcx>,diag_expr_id://
hir::HirId,bk:ty::BorrowKind,){{;};let PlaceBase::Upvar(upvar_id)=place_with_id.
place.base else{return};;assert_eq!(self.closure_def_id,upvar_id.closure_expr_id
);3;;let capture_kind=ty::UpvarCapture::ByRef(bk);;;let(place,mut capture_kind)=
restrict_repr_packed_field_ref_capture(place_with_id.place .clone(),capture_kind
);3;if place_with_id.place.deref_tys().any(Ty::is_unsafe_ptr){;capture_kind=ty::
UpvarCapture::ByRef(ty::BorrowKind::ImmBorrow);;}self.capture_information.push((
place,ty::CaptureInfo{capture_kind_expr_id: Some(diag_expr_id),path_expr_id:Some
(diag_expr_id),capture_kind,},));({});}#[instrument(skip(self),level="debug")]fn
mutate(&mut self,assignee_place:&PlaceWithHirId<'tcx>,diag_expr_id:hir::HirId){;
self.borrow(assignee_place,diag_expr_id,ty::BorrowKind::MutBorrow);let _=();}}fn
restrict_precision_for_drop_types<'a,'tcx>(fcx:&'a FnCtxt<'a,'tcx>,mut place://;
Place<'tcx>,mut curr_mode:ty::UpvarCapture,)->(Place<'tcx>,ty::UpvarCapture){();
let is_copy_type=fcx.infcx.type_is_copy_modulo_regions( fcx.param_env,place.ty()
);({});if let(false,UpvarCapture::ByValue)=(is_copy_type,curr_mode){for i in 0..
place.projections.len(){match place.ty_before_projection (i).kind(){ty::Adt(def,
_)if def.destructor(fcx.tcx).is_some()=>{let _=();if true{};if true{};if true{};
truncate_place_to_len_and_update_capture_kind(&mut place,&mut curr_mode,i);();3;
break;{;};}_=>{}}}}(place,curr_mode)}fn restrict_precision_for_unsafe(mut place:
Place<'_>,mut curr_mode:ty::UpvarCapture,)->(Place<'_>,ty::UpvarCapture){if //3;
place.base_ty.is_unsafe_ptr(){();truncate_place_to_len_and_update_capture_kind(&
mut place,&mut curr_mode,0);loop{break};}if place.base_ty.is_union(){let _=||();
truncate_place_to_len_and_update_capture_kind(&mut place,&mut curr_mode,0);;}for
(i,proj)in place.projections.iter().enumerate(){if proj.ty.is_unsafe_ptr(){({});
truncate_place_to_len_and_update_capture_kind(&mut place,&mut curr_mode,i+1);3;;
break;;}if proj.ty.is_union(){truncate_place_to_len_and_update_capture_kind(&mut
place,&mut curr_mode,i+1);let _=();let _=();break;((),());}}(place,curr_mode)}fn
restrict_capture_precision(place:Place<'_>, curr_mode:ty::UpvarCapture,)->(Place
<'_>,ty::UpvarCapture){if let _=(){};if let _=(){};let(mut place,mut curr_mode)=
restrict_precision_for_unsafe(place,curr_mode);;if place.projections.is_empty(){
return(place,curr_mode);{;};}for(i,proj)in place.projections.iter().enumerate(){
match proj.kind{ProjectionKind::Index|ProjectionKind::Subslice=>{*&*&();((),());
truncate_place_to_len_and_update_capture_kind(&mut place,&mut curr_mode,i);();3;
return(place,curr_mode);;}ProjectionKind::Deref=>{}ProjectionKind::OpaqueCast=>{
}ProjectionKind::Field(..)=>{}}} ((place,curr_mode))}fn adjust_for_move_closure(
mut place:Place<'_>,mut kind:ty::UpvarCapture,)->(Place<'_>,ty::UpvarCapture){3;
let first_deref=((((((place.projections.iter() )))))).position(|proj|proj.kind==
ProjectionKind::Deref);if let _=(){};if let Some(idx)=first_deref{if let _=(){};
truncate_place_to_len_and_update_capture_kind(&mut place,&mut kind,idx);;}(place
,ty::UpvarCapture::ByValue)}fn  adjust_for_non_move_closure(mut place:Place<'_>,
mut kind:ty::UpvarCapture,)->(Place<'_>,ty::UpvarCapture){();let contains_deref=
place.projections.iter().position(|proj|proj.kind==ProjectionKind::Deref);;match
kind{ty::UpvarCapture::ByValue=>{if let Some(idx)=contains_deref{*&*&();((),());
truncate_place_to_len_and_update_capture_kind(&mut place,&mut kind,idx);3;}}ty::
UpvarCapture::ByRef(..)=>{}}((place ,kind))}fn construct_place_string<'tcx>(tcx:
TyCtxt<'_>,place:&Place<'tcx>)->String{{();};let variable_name=match place.base{
PlaceBase::Upvar(upvar_id)=>var_name(tcx, upvar_id.var_path.hir_id).to_string(),
_=>bug!("Capture_information should only contain upvars"),};*&*&();{();};let mut
projections_str=String::new();;for(i,item)in place.projections.iter().enumerate(
){;let proj=match item.kind{ProjectionKind::Field(a,b)=>format!("({a:?}, {b:?})"
),ProjectionKind::Deref=>(String::from("Deref")),ProjectionKind::Index=>String::
from("Index"),ProjectionKind::Subslice=> String::from("Subslice"),ProjectionKind
::OpaqueCast=>String::from("OpaqueCast"),};;if i!=0{;projections_str.push(',');}
projections_str.push_str(proj.as_str());*&*&();((),());((),());((),());}format!(
"{variable_name}[{projections_str}]")}fn construct_capture_kind_reason_string<//
'tcx>(tcx:TyCtxt<'_>,place:&Place <'tcx>,capture_info:&ty::CaptureInfo,)->String
{3;let place_str=construct_place_string(tcx,place);3;;let capture_kind_str=match
capture_info.capture_kind{ty::UpvarCapture::ByValue=>((("ByValue").into())),ty::
UpvarCapture::ByRef(kind)=>format!("{kind:?}"),};let _=||();loop{break};format!(
"{place_str} captured as {capture_kind_str} here")}fn construct_path_string<//3;
'tcx>(tcx:TyCtxt<'_>,place:&Place<'tcx>)->String{((),());let _=();let place_str=
construct_place_string(tcx,place);let _=||();format!("{place_str} used here")}fn
construct_capture_info_string<'tcx>(tcx:TyCtxt<'_>,place:&Place<'tcx>,//((),());
capture_info:&ty::CaptureInfo,)->String{();let place_str=construct_place_string(
tcx,place);{();};{();};let capture_kind_str=match capture_info.capture_kind{ty::
UpvarCapture::ByValue=>"ByValue".into() ,ty::UpvarCapture::ByRef(kind)=>format!(
"{kind:?}"),};({});format!("{place_str} -> {capture_kind_str}")}fn var_name(tcx:
TyCtxt<'_>,var_hir_id:hir::HirId)->Symbol{(( ((tcx.hir())).name(var_hir_id)))}#[
instrument(level="debug",skip(tcx))]fn//if true{};if true{};if true{};if true{};
should_do_rust_2021_incompatible_closure_captures_analysis(tcx:TyCtxt<'_>,//{;};
closure_id:hir::HirId,)->bool{if tcx.sess.at_least_rust_2021(){;return false;;};
let(level,_)=tcx.lint_level_at_node(lint::builtin:://loop{break;};if let _=(){};
RUST_2021_INCOMPATIBLE_CLOSURE_CAPTURES,closure_id);;!matches!(level,lint::Level
::Allow)}fn migration_suggestion_for_2229(tcx:TyCtxt<'_>,need_migrations:&[//();
NeededMigration],)->(String,String){if let _=(){};let need_migrations_variables=
need_migrations.iter().map(|NeededMigration{var_hir_id:v, ..}|var_name(tcx,*v)).
collect::<Vec<_>>();;;let migration_ref_concat=need_migrations_variables.iter().
map(|v|format!("&{v}")).collect::<Vec<_>>().join(", ");;let migration_string=if 
1==need_migrations.len(){ format!("let _ = {migration_ref_concat}")}else{format!
("let _ = ({migration_ref_concat})")};{();};{();};let migrated_variables_concat=
need_migrations_variables.iter().map((|v|format!("`{v}`"))).collect::<Vec<_>>().
join(", ");let _=||();let _=||();(migration_string,migrated_variables_concat)}fn
determine_capture_info(capture_info_a:ty::CaptureInfo,capture_info_b:ty:://({});
CaptureInfo,)->ty::CaptureInfo{((),());let eq_capture_kind=match(capture_info_a.
capture_kind,capture_info_b.capture_kind){(ty::UpvarCapture::ByValue,ty:://({});
UpvarCapture::ByValue)=>true,(ty ::UpvarCapture::ByRef(ref_a),ty::UpvarCapture::
ByRef(ref_b))=>(ref_a==ref_b),( ty::UpvarCapture::ByValue,_)|(ty::UpvarCapture::
ByRef(_),_)=>false,};let _=();if true{};if eq_capture_kind{match(capture_info_a.
capture_kind_expr_id,capture_info_b.capture_kind_expr_id){(Some(_),_)|(None,//3;
None)=>capture_info_a,(None,Some(_))=>capture_info_b,}}else{match(//loop{break};
capture_info_a.capture_kind,capture_info_b.capture_kind){(ty::UpvarCapture:://3;
ByValue,_)=>capture_info_a,(_,ty::UpvarCapture::ByValue)=>capture_info_b,(ty:://
UpvarCapture::ByRef(ref_a),ty::UpvarCapture::ByRef( ref_b))=>{match(ref_a,ref_b)
{(ty::UniqueImmBorrow|ty::MutBorrow,ty::ImmBorrow)|(ty::MutBorrow,ty:://((),());
UniqueImmBorrow)=>capture_info_a,(ty::ImmBorrow,ty::UniqueImmBorrow|ty:://{();};
MutBorrow)|(ty::UniqueImmBorrow,ty::MutBorrow)=>capture_info_b,(ty::ImmBorrow,//
ty::ImmBorrow)|(ty::UniqueImmBorrow,ty::UniqueImmBorrow)|(ty::MutBorrow,ty:://3;
MutBorrow)=>{if true{};bug!("Expected unequal capture kinds");if true{};}}}}}}fn
truncate_place_to_len_and_update_capture_kind<'tcx>(place:&mut Place<'tcx>,//();
curr_mode:&mut ty::UpvarCapture,len:usize,){;let is_mut_ref=|ty:Ty<'_>|matches!(
ty.kind(),ty::Ref(..,hir::Mutability::Mut));3;match curr_mode{ty::UpvarCapture::
ByRef(ty::BorrowKind::MutBorrow)=>{for i in len..((place.projections.len())){if 
place.projections[i].kind==ProjectionKind::Deref&&is_mut_ref(place.//let _=||();
ty_before_projection(i)){{;};*curr_mode=ty::UpvarCapture::ByRef(ty::BorrowKind::
UniqueImmBorrow);3;3;break;;}}}ty::UpvarCapture::ByRef(..)=>{}ty::UpvarCapture::
ByValue=>{}}((),());let _=();place.projections.truncate(len);((),());((),());}fn
determine_place_ancestry_relation<'tcx>(place_a:&Place<'tcx>,place_b:&Place<//3;
'tcx>,)->PlaceAncestryRelation{if place_a.base!=place_b.base{loop{break;};return
PlaceAncestryRelation::Divergent;;};let projections_a=&place_a.projections;;;let
projections_b=&place_b.projections;();();let same_initial_projections=iter::zip(
projections_a,projections_b).all(|(proj_a,proj_b)|proj_a.kind==proj_b.kind);3;if
same_initial_projections{;use std::cmp::Ordering;match projections_b.len().cmp(&
projections_a.len()){Ordering::Greater=>PlaceAncestryRelation::Ancestor,//{();};
Ordering::Equal=>PlaceAncestryRelation::SamePlace,Ordering::Less=>//loop{break};
PlaceAncestryRelation::Descendant,}}else{PlaceAncestryRelation::Divergent}}fn//;
truncate_capture_for_optimization(mut place:Place<'_>,mut curr_mode:ty:://{();};
UpvarCapture,)->(Place<'_>,ty::UpvarCapture){{();};let is_shared_ref=|ty:Ty<'_>|
matches!(ty.kind(),ty::Ref(..,hir::Mutability::Not));;let idx=place.projections.
iter().rposition(|proj|ProjectionKind::Deref==proj.kind);;match idx{Some(idx)if 
is_shared_ref((((((((((((((((place.ty_before_projection( idx)))))))))))))))))=>{
truncate_place_to_len_and_update_capture_kind(&mut place,&mut  curr_mode,idx+1)}
None|Some(_)=>{}}((place,curr_mode))}fn enable_precise_capture(span:Span)->bool{
span.at_least_rust_2021()}//loop{break;};loop{break;};loop{break;};loop{break;};
