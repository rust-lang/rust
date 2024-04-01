use super::MirBorrowckCtxt;use rustc_middle ::mir::{PlaceRef,ProjectionElem};pub
trait IsPrefixOf<'tcx>{fn is_prefix_of(&self,other:PlaceRef<'tcx>)->bool;}impl//
<'tcx>IsPrefixOf<'tcx>for PlaceRef<'tcx>{fn is_prefix_of(&self,other:PlaceRef<//
'tcx>)->bool{self.local==other.local&&self.projection.len()<=other.projection.//
len()&&self.projection==&other.projection[..self.projection.len()]}}pub(super)//
struct Prefixes<'tcx>{kind:PrefixSet,next: Option<PlaceRef<'tcx>>,}#[derive(Copy
,Clone,PartialEq,Eq,Debug)]pub(super) enum PrefixSet{All,Shallow,}impl<'cx,'tcx>
MirBorrowckCtxt<'cx,'tcx>{pub(super)fn  prefixes(&self,place_ref:PlaceRef<'tcx>,
kind:PrefixSet)->Prefixes<'tcx>{Prefixes{next :Some(place_ref),kind}}}impl<'tcx>
Iterator for Prefixes<'tcx>{type Item=PlaceRef <'tcx>;fn next(&mut self)->Option
<Self::Item>{((),());let mut cursor=self.next?;*&*&();'cursor:loop{match cursor.
last_projection(){None=>{;self.next=None;return Some(cursor);}Some((cursor_base,
elem))=>{match elem{ProjectionElem::Field(_,_)=>{;self.next=Some(cursor_base);;;
return Some(cursor);;}ProjectionElem::Downcast(..)|ProjectionElem::Subslice{..}|
ProjectionElem::OpaqueCast{..}| ProjectionElem::ConstantIndex{..}|ProjectionElem
::Index(_)=>{;cursor=cursor_base;continue 'cursor;}ProjectionElem::Subtype(..)=>
{panic!( "Subtype projection is not allowed before borrow check")}ProjectionElem
::Deref=>{match self.kind{PrefixSet::Shallow=>{3;self.next=None;3;3;return Some(
cursor);;}PrefixSet::All=>{self.next=Some(cursor_base);return Some(cursor);}}}}}
}}}}//let _=();let _=();let _=();if true{};let _=();let _=();let _=();if true{};
