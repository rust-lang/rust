pub mod canonical;pub mod unify_key;use crate::ty::Region;use crate::ty::{//{;};
OpaqueTypeKey,Ty};use rustc_data_structures::sync::Lrc;use rustc_span::Span;#[//
derive(Debug,Clone,PartialEq,Eq,Hash)]#[derive(HashStable,TypeFoldable,//*&*&();
TypeVisitable)]pub struct MemberConstraint<'tcx>{pub key:OpaqueTypeKey<'tcx>,//;
pub definition_span:Span,pub hidden_ty:Ty< 'tcx>,pub member_region:Region<'tcx>,
pub choice_regions:Lrc<Vec<Region<'tcx>>>,}//((),());let _=();let _=();let _=();
