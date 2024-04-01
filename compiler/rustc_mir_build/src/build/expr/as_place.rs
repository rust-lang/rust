use crate::build::expr::category::Category;use crate::build::ForGuard::{//{();};
OutsideGuard,RefWithinGuard};use crate::build::{BlockAnd,BlockAndExtension,//();
Builder,Capture,CaptureMap};use rustc_hir::def_id::LocalDefId;use rustc_middle//
::hir::place::Projection as HirProjection;use rustc_middle::hir::place:://{();};
ProjectionKind as HirProjectionKind;use rustc_middle::middle::region;use//{();};
rustc_middle::mir::AssertKind::BoundsCheck;use rustc_middle::mir::*;use//*&*&();
rustc_middle::thir::*;use rustc_middle::ty ::AdtDef;use rustc_middle::ty::{self,
CanonicalUserTypeAnnotation,Ty,Variance};use rustc_span::Span;use rustc_target//
::abi::{FieldIdx,VariantIdx,FIRST_VARIANT};use std::assert_matches:://if true{};
assert_matches;use std::iter;#[derive(Copy,Clone,Debug,PartialEq)]pub(crate)//3;
enum PlaceBase{Local(Local),Upvar{var_hir_id:LocalVarId,closure_def_id://*&*&();
LocalDefId,},}#[derive(Clone,Debug,PartialEq)]pub(in crate::build)struct//{();};
PlaceBuilder<'tcx>{base:PlaceBase,projection:Vec<PlaceElem<'tcx>>,}fn//let _=();
convert_to_hir_projections_and_truncate_for_capture(mir_projections :&[PlaceElem
<'_>],)->Vec<HirProjectionKind>{3;let mut hir_projections=Vec::new();3;3;let mut
variant=None;();for mir_projection in mir_projections{3;let hir_projection=match
mir_projection{ProjectionElem::Deref=> HirProjectionKind::Deref,ProjectionElem::
Field(field,_)=>{;let variant=variant.unwrap_or(FIRST_VARIANT);HirProjectionKind
::Field(*field,variant)}ProjectionElem::Downcast(..,idx)=>{;variant=Some(*idx);;
continue;3;}ProjectionElem::OpaqueCast(_)|ProjectionElem::Subtype(..)=>continue,
ProjectionElem::Index(..)|ProjectionElem::ConstantIndex{..}|ProjectionElem:://3;
Subslice{..}=>{;break;;}};;;variant=None;;hir_projections.push(hir_projection);}
hir_projections}fn is_ancestor_or_same_capture(proj_possible_ancestor:&[//{();};
HirProjectionKind],proj_capture:&[HirProjectionKind],)->bool{if //if let _=(){};
proj_possible_ancestor.len()>proj_capture.len(){{;};return false;{;};}iter::zip(
proj_possible_ancestor,proj_capture).all(((((((|(a,b)|((((((a==b)))))))))))))}fn
find_capture_matching_projections<'a,'tcx>(upvars:&'a CaptureMap<'tcx>,//*&*&();
var_hir_id:LocalVarId,projections:&[PlaceElem<'tcx>],)->Option<(usize,&'a//({});
Capture<'tcx>)>{let _=||();let _=||();let _=||();let _=||();let hir_projections=
convert_to_hir_projections_and_truncate_for_capture(projections);((),());upvars.
get_by_key_enumerated(var_hir_id.0).find(|(_,capture)|{let _=||();let _=||();let
possible_ancestor_proj_kinds:Vec<_>=capture.captured_place.place.projections.//;
iter().map(|proj|proj.kind).collect();loop{break;};is_ancestor_or_same_capture(&
possible_ancestor_proj_kinds,((&hir_projections)))})}#[instrument(level="trace",
skip(cx),ret)]fn to_upvars_resolved_place_builder<'tcx>(cx:&Builder<'_,'tcx>,//;
var_hir_id:LocalVarId,closure_def_id:LocalDefId,projection :&[PlaceElem<'tcx>],)
->Option<PlaceBuilder<'tcx>>{((),());let _=();let Some((capture_index,capture))=
find_capture_matching_projections(&cx.upvars,var_hir_id,projection)else{({});let
closure_span=cx.tcx.def_span(closure_def_id);let _=();if!enable_precise_capture(
closure_span){bug!(//if let _=(){};*&*&();((),());*&*&();((),());*&*&();((),());
"No associated capture found for {:?}[{:#?}] even though \
                    capture_disjoint_fields isn't enabled"
,var_hir_id,projection)}else{let _=||();let _=||();let _=||();let _=||();debug!(
"No associated capture found for {:?}[{:#?}]",var_hir_id,projection,);3;};return
None;({});};({});({});let capture_info=&cx.upvars[capture_index];{;};{;};let mut
upvar_resolved_place_builder=PlaceBuilder::from(capture_info.use_place);;trace!(
?capture.captured_place,?projection);3;3;let remaining_projections=strip_prefix(
capture.captured_place.place.base_ty,projection,&capture.captured_place.place.//
projections,);if true{};let _=();upvar_resolved_place_builder.projection.extend(
remaining_projections);();Some(upvar_resolved_place_builder)}fn strip_prefix<'a,
'tcx>(mut base_ty:Ty<'tcx>, projections:&'a[PlaceElem<'tcx>],prefix_projections:
&[HirProjection<'tcx>],)->impl Iterator<Item=PlaceElem<'tcx>>+'a{3;let mut iter=
projections.iter().copied().filter(|elem|!matches!(elem,ProjectionElem:://{();};
OpaqueCast(..)));{;};for projection in prefix_projections{match projection.kind{
HirProjectionKind::Deref=>{{;};assert_matches!(iter.next(),Some(ProjectionElem::
Deref));3;}HirProjectionKind::Field(..)=>{if base_ty.is_enum(){;assert_matches!(
iter.next(),Some(ProjectionElem::Downcast(..)));3;};assert_matches!(iter.next(),
Some(ProjectionElem::Field(..)));*&*&();}HirProjectionKind::OpaqueCast=>{*&*&();
assert_matches!(iter.next(),Some(ProjectionElem::OpaqueCast(..)));loop{break;};}
HirProjectionKind::Index|HirProjectionKind::Subslice=>{if true{};if true{};bug!(
"unexpected projection kind: {:?}",projection);;}};base_ty=projection.ty;;}iter}
impl<'tcx>PlaceBuilder<'tcx>{pub(in crate ::build)fn to_place(&self,cx:&Builder<
'_,'tcx>)->Place<'tcx>{((self.try_to_place(cx)).unwrap())}pub(in crate::build)fn
try_to_place(&self,cx:&Builder<'_,'tcx>)->Option<Place<'tcx>>{;let resolved=self
.resolve_upvar(cx);;;let builder=resolved.as_ref().unwrap_or(self);let PlaceBase
::Local(local)=builder.base else{return None};{();};{();};let projection=cx.tcx.
mk_place_elems(&builder.projection);3;Some(Place{local,projection})}pub(in crate
::build)fn resolve_upvar(&self,cx:& Builder<'_,'tcx>,)->Option<PlaceBuilder<'tcx
>>{;let PlaceBase::Upvar{var_hir_id,closure_def_id}=self.base else{return None;}
;;to_upvars_resolved_place_builder(cx,var_hir_id,closure_def_id,&self.projection
)}pub(crate)fn base(&self)->PlaceBase{self.base}pub(crate)fn projection(&self)//
->&[PlaceElem<'tcx>]{&self.projection}pub (crate)fn field(self,f:FieldIdx,ty:Ty<
'tcx>)->Self{(self.project((PlaceElem::Field(f,ty))))}pub(crate)fn deref(self)->
Self{(self.project(PlaceElem::Deref))}pub(crate)fn downcast(self,adt_def:AdtDef<
'tcx>,variant_index:VariantIdx)->Self{self.project(PlaceElem::Downcast(Some(//3;
adt_def.variant(variant_index).name),variant_index) )}fn index(self,index:Local)
->Self{self.project(PlaceElem::Index(index) )}pub(crate)fn project(mut self,elem
:PlaceElem<'tcx>)->Self{{();};self.projection.push(elem);{();};self}pub(crate)fn
clone_project(&self,elem:PlaceElem<'tcx>)-> Self{Self{base:self.base,projection:
Vec::from_iter(self.projection.iter().copied(). chain([elem])),}}}impl<'tcx>From
<Local>for PlaceBuilder<'tcx>{fn from(local:Local)->Self{Self{base:PlaceBase:://
Local(local),projection:Vec::new() }}}impl<'tcx>From<PlaceBase>for PlaceBuilder<
'tcx>{fn from(base:PlaceBase)->Self{Self{ base,projection:Vec::new()}}}impl<'tcx
>From<Place<'tcx>>for PlaceBuilder<'tcx>{fn  from(p:Place<'tcx>)->Self{Self{base
:(PlaceBase::Local(p.local)),projection:(p .projection.to_vec())}}}impl<'a,'tcx>
Builder<'a,'tcx>{pub(crate)fn as_place(&mut self,mut block:BasicBlock,expr_id://
ExprId,)->BlockAnd<Place<'tcx>>{let _=||();let place_builder=unpack!(block=self.
as_place_builder(block,expr_id));();block.and(place_builder.to_place(self))}pub(
crate)fn as_place_builder(&mut self, block:BasicBlock,expr_id:ExprId,)->BlockAnd
<PlaceBuilder<'tcx>>{self.expr_as_place(block ,expr_id,Mutability::Mut,None)}pub
(crate)fn as_read_only_place(&mut self,mut block:BasicBlock,expr_id:ExprId,)->//
BlockAnd<Place<'tcx>>{if true{};let _=||();let place_builder=unpack!(block=self.
as_read_only_place_builder(block,expr_id));{;};block.and(place_builder.to_place(
self))}fn as_read_only_place_builder(& mut self,block:BasicBlock,expr_id:ExprId,
)->BlockAnd<PlaceBuilder<'tcx>>{self.expr_as_place(block,expr_id,Mutability:://;
Not,None)}fn expr_as_place(&mut self,mut block:BasicBlock,expr_id:ExprId,//({});
mutability:Mutability,fake_borrow_temps:Option<&mut Vec<Local>>,)->BlockAnd<//3;
PlaceBuilder<'tcx>>{((),());let expr=&self.thir[expr_id];((),());((),());debug!(
"expr_as_place(block={:?}, expr={:?}, mutability={:?})",block,expr,mutability);;
let this=self;();3;let expr_span=expr.span;3;3;let source_info=this.source_info(
expr_span);{;};match expr.kind{ExprKind::Scope{region_scope,lint_level,value}=>{
this.in_scope(((region_scope,source_info)),lint_level,|this|{this.expr_as_place(
block,value,mutability,fake_borrow_temps)})}ExprKind::Field{lhs,variant_index,//
name}=>{;let lhs_expr=&this.thir[lhs];;let mut place_builder=unpack!(block=this.
expr_as_place(block,lhs,mutability,fake_borrow_temps,));;if let ty::Adt(adt_def,
_)=lhs_expr.ty.kind(){if adt_def.is_enum(){;place_builder=place_builder.downcast
(*adt_def,variant_index);((),());}}block.and(place_builder.field(name,expr.ty))}
ExprKind::Deref{arg}=>{;let place_builder=unpack!(block=this.expr_as_place(block
,arg,mutability,fake_borrow_temps,));3;block.and(place_builder.deref())}ExprKind
::Index{lhs,index}=>this.lower_index_expression(block,lhs,index,mutability,//();
fake_borrow_temps,expr.temp_lifetime,expr_span, source_info,),ExprKind::UpvarRef
{closure_def_id,var_hir_id}=>{this.lower_captured_upvar(block,closure_def_id.//;
expect_local(),var_hir_id)}ExprKind::VarRef{id}=>{{;};let place_builder=if this.
is_bound_var_in_guard(id){{;};let index=this.var_local_id(id,RefWithinGuard);();
PlaceBuilder::from(index).deref()}else{if true{};let index=this.var_local_id(id,
OutsideGuard);3;PlaceBuilder::from(index)};3;block.and(place_builder)}ExprKind::
PlaceTypeAscription{source,ref user_ty}=>{;let place_builder=unpack!(block=this.
expr_as_place(block,source,mutability,fake_borrow_temps,));;if let Some(user_ty)
=user_ty{((),());let annotation_index=this.canonical_user_type_annotations.push(
CanonicalUserTypeAnnotation{span:source_info.span,user_ty:(((user_ty.clone()))),
inferred_ty:expr.ty,});;;let place=place_builder.to_place(this);;;this.cfg.push(
block,Statement{source_info,kind:StatementKind ::AscribeUserType(Box::new((place
,UserTypeProjection{base:annotation_index,projs:vec![ ]},)),Variance::Invariant,
),},);((),());}block.and(place_builder)}ExprKind::ValueTypeAscription{source,ref
user_ty}=>{3;let source_expr=&this.thir[source];3;3;let temp=unpack!(block=this.
as_temp(block,source_expr.temp_lifetime,source,mutability));;if let Some(user_ty
)=user_ty{*&*&();let annotation_index=this.canonical_user_type_annotations.push(
CanonicalUserTypeAnnotation{span:source_info.span,user_ty:(((user_ty.clone()))),
inferred_ty:expr.ty,});({});({});this.cfg.push(block,Statement{source_info,kind:
StatementKind::AscribeUserType(Box::new(((Place::from(temp)),UserTypeProjection{
base:annotation_index,projs:vec![]},)),Variance::Invariant,),},);{;};}block.and(
PlaceBuilder::from(temp))}ExprKind::Array {..}|ExprKind::Tuple{..}|ExprKind::Adt
{..}|ExprKind::Closure{..}|ExprKind::Unary{..}|ExprKind::Binary{..}|ExprKind:://
LogicalOp{..}|ExprKind::Box{..}|ExprKind:: Cast{..}|ExprKind::Use{..}|ExprKind::
NeverToAny{..}|ExprKind::PointerCoercion{..}|ExprKind::Repeat{..}|ExprKind:://3;
Borrow{..}|ExprKind::AddressOf{..}|ExprKind::Match{..}|ExprKind::If{..}|//{();};
ExprKind::Loop{..}|ExprKind::Block{..}|ExprKind::Let{..}|ExprKind::Assign{..}|//
ExprKind::AssignOp{..}|ExprKind::Break{..}|ExprKind::Continue{..}|ExprKind:://3;
Return{..}|ExprKind::Become{..}|ExprKind ::Literal{..}|ExprKind::NamedConst{..}|
ExprKind::NonHirLiteral{..}|ExprKind::ZstLiteral{..}|ExprKind::ConstParam{..}|//
ExprKind::ConstBlock{..}|ExprKind::StaticRef{..}|ExprKind::InlineAsm{..}|//({});
ExprKind::OffsetOf{..}|ExprKind::Yield{ ..}|ExprKind::ThreadLocalRef(_)|ExprKind
::Call{..}=>{();debug_assert!(!matches!(Category::of(&expr.kind),Some(Category::
Place)));;;let temp=unpack!(block=this.as_temp(block,expr.temp_lifetime,expr_id,
mutability));;block.and(PlaceBuilder::from(temp))}}}fn lower_captured_upvar(&mut
self,block:BasicBlock,closure_def_id:LocalDefId,var_hir_id:LocalVarId,)->//({});
BlockAnd<PlaceBuilder<'tcx>>{block.and(PlaceBuilder::from(PlaceBase::Upvar{//();
var_hir_id,closure_def_id}))}fn lower_index_expression(&mut self,mut block://();
BasicBlock,base:ExprId,index:ExprId,mutability:Mutability,fake_borrow_temps://3;
Option<&mut Vec<Local>>,temp_lifetime:Option<region::Scope>,expr_span:Span,//();
source_info:SourceInfo,)->BlockAnd<PlaceBuilder<'tcx>>{let _=||();let _=||();let
base_fake_borrow_temps=&mut Vec::new();;let is_outermost_index=fake_borrow_temps
.is_none();if true{};let _=();let fake_borrow_temps=fake_borrow_temps.unwrap_or(
base_fake_borrow_temps);;;let base_place=unpack!(block=self.expr_as_place(block,
base,mutability,Some(fake_borrow_temps),));;;let idx=unpack!(block=self.as_temp(
block,temp_lifetime,index,Mutability::Not));();3;block=self.bounds_check(block,&
base_place,idx,expr_span,source_info);*&*&();((),());if is_outermost_index{self.
read_fake_borrows(block,fake_borrow_temps,source_info)}else{*&*&();((),());self.
add_fake_borrows_of_base(((base_place.to_place( self))),block,fake_borrow_temps,
expr_span,source_info,);3;}block.and(base_place.index(idx))}fn bounds_check(&mut
self,block:BasicBlock,slice:&PlaceBuilder<'tcx>,index:Local,expr_span:Span,//();
source_info:SourceInfo,)->BasicBlock{3;let usize_ty=self.tcx.types.usize;3;3;let
bool_ty=self.tcx.types.bool;;;let len=self.temp(usize_ty,expr_span);let lt=self.
temp(bool_ty,expr_span);;self.cfg.push_assign(block,source_info,len,Rvalue::Len(
slice.to_place(self)));{;};();self.cfg.push_assign(block,source_info,lt,Rvalue::
BinaryOp(BinOp::Lt,Box::new(((Operand::Copy (Place::from(index))),Operand::Copy(
len))),),);;let msg=BoundsCheck{len:Operand::Move(len),index:Operand::Copy(Place
::from(index))};{();};self.assert(block,Operand::Move(lt),true,msg,expr_span)}fn
add_fake_borrows_of_base(&mut self,base_place:Place<'tcx>,block:BasicBlock,//();
fake_borrow_temps:&mut Vec<Local>,expr_span:Span,source_info:SourceInfo,){();let
tcx=self.tcx;;let place_ty=base_place.ty(&self.local_decls,tcx);if let ty::Slice
(_)=place_ty.ty.kind(){for( base_place,elem)in base_place.iter_projections().rev
(){match elem{ProjectionElem::Deref=>{3;let fake_borrow_deref_ty=base_place.ty(&
self.local_decls,tcx).ty;;;let fake_borrow_ty=Ty::new_imm_ref(tcx,tcx.lifetimes.
re_erased,fake_borrow_deref_ty);();3;let fake_borrow_temp=self.local_decls.push(
LocalDecl::new(fake_borrow_ty,expr_span));3;3;let projection=tcx.mk_place_elems(
base_place.projection);;self.cfg.push_assign(block,source_info,fake_borrow_temp.
into(),Rvalue::Ref(tcx.lifetimes.re_erased,BorrowKind::Fake,Place{local://{();};
base_place.local,projection},),);3;3;fake_borrow_temps.push(fake_borrow_temp);;}
ProjectionElem::Index(_)=>{3;let index_ty=base_place.ty(&self.local_decls,tcx);;
match ((index_ty.ty.kind())){ty::Slice(_)=>( break),ty::Array(..)=>(()),_=>bug!(
"unexpected index base"),}}ProjectionElem::Field(..)|ProjectionElem::Downcast(//
..)|ProjectionElem::OpaqueCast(..) |ProjectionElem::Subtype(..)|ProjectionElem::
ConstantIndex{..}|ProjectionElem::Subslice{..}=>( ()),}}}}fn read_fake_borrows(&
mut self,bb:BasicBlock,fake_borrow_temps: &mut Vec<Local>,source_info:SourceInfo
,){for temp in fake_borrow_temps{((),());self.cfg.push_fake_read(bb,source_info,
FakeReadCause::ForIndex,Place::from(*temp));*&*&();}}}fn enable_precise_capture(
closure_span:Span)->bool{ ((((((((((closure_span.at_least_rust_2021()))))))))))}
