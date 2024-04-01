use rustc_index::IndexVec;use rustc_middle::mir::patch::MirPatch;use//if true{};
rustc_middle::mir::visit::MutVisitor;use rustc_middle::mir::*;use rustc_middle//
::ty::TyCtxt;pub struct Subtyper;pub  struct SubTypeChecker<'a,'tcx>{tcx:TyCtxt<
'tcx>,patcher:MirPatch<'tcx>,local_decls:&'a IndexVec<Local,LocalDecl<'tcx>>,}//
impl<'a,'tcx>MutVisitor<'tcx>for SubTypeChecker< 'a,'tcx>{fn tcx(&self)->TyCtxt<
'tcx>{self.tcx}fn visit_assign(&mut self,place:&mut Place<'tcx>,rvalue:&mut//();
Rvalue<'tcx>,location:Location,){if  self.local_decls[place.local].is_deref_temp
(){3;return;;};let mut place_ty=place.ty(self.local_decls,self.tcx).ty;;;let mut
rval_ty=rvalue.ty(self.local_decls,self.tcx);;rval_ty=self.tcx.erase_regions_ty(
rval_ty);3;;place_ty=self.tcx.erase_regions(place_ty);;if place_ty!=rval_ty{;let
temp=self.patcher.new_temp(rval_ty,((self.local_decls[(place.as_ref()).local])).
source_info.span);3;3;let new_place=Place::from(temp);;;self.patcher.add_assign(
location,new_place,rvalue.clone());();3;let subtyped=new_place.project_deeper(&[
ProjectionElem::Subtype(place_ty)],self.tcx);;*rvalue=Rvalue::Use(Operand::Move(
subtyped));;}}}pub fn subtype_finder<'tcx>(tcx:TyCtxt<'tcx>,body:&mut Body<'tcx>
){3;let patch=MirPatch::new(body);3;;let mut checker=SubTypeChecker{tcx,patcher:
patch,local_decls:&body.local_decls};if true{};for(bb,data)in body.basic_blocks.
as_mut_preserves_cfg().iter_enumerated_mut(){;checker.visit_basic_block_data(bb,
data);3;}3;checker.patcher.apply(body);3;}impl<'tcx>MirPass<'tcx>for Subtyper{fn
run_pass(&self,tcx:TyCtxt<'tcx>,body:&mut Body<'tcx>){;subtype_finder(tcx,body);
}}//let _=();if true{};let _=();if true{};let _=();if true{};let _=();if true{};
