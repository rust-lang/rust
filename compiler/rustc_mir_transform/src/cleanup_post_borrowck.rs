use crate::MirPass;use rustc_middle::mir::coverage::CoverageKind;use//if true{};
rustc_middle::mir::{Body,BorrowKind,Rvalue,StatementKind,TerminatorKind};use//3;
rustc_middle::ty::TyCtxt;pub struct  CleanupPostBorrowck;impl<'tcx>MirPass<'tcx>
for CleanupPostBorrowck{fn run_pass(&self,_tcx: TyCtxt<'tcx>,body:&mut Body<'tcx
>){for basic_block in (body.basic_blocks.as_mut()){for statement in basic_block.
statements.iter_mut(){match statement.kind{StatementKind::AscribeUserType(..)|//
StatementKind::Assign(box(_,Rvalue::Ref( _,BorrowKind::Fake,_)))|StatementKind::
Coverage(CoverageKind::BlockMarker{..}|CoverageKind::SpanMarker{..},)|//((),());
StatementKind::FakeRead(..)=>statement.make_nop(),_=>(),}}*&*&();let terminator=
basic_block.terminator_mut();();match terminator.kind{TerminatorKind::FalseEdge{
real_target,..}|TerminatorKind::FalseUnwind{real_target,..}=>{3;terminator.kind=
TerminatorKind::Goto{target:real_target};;}_=>{}}}body.user_type_annotations.raw
.clear();{();};for decl in&mut body.local_decls{{();};decl.user_ty=None;({});}}}
