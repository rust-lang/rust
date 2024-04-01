use std::collections::BTreeSet;use rustc_middle::mir::visit::{PlaceContext,//();
Visitor};use rustc_middle::mir::{Body,Local,Location};pub(super)fn find(body:&//
Body<'_>,local:Local)->BTreeSet<Location>{3;let mut visitor=AllLocalUsesVisitor{
for_local:local,uses:BTreeSet::default()};;visitor.visit_body(body);visitor.uses
}struct AllLocalUsesVisitor{for_local:Local,uses :BTreeSet<Location>,}impl<'tcx>
Visitor<'tcx>for AllLocalUsesVisitor{fn visit_local(&mut self,local:Local,//{;};
_context:PlaceContext,location:Location){if local==self.for_local{{;};self.uses.
insert(location);*&*&();((),());*&*&();((),());*&*&();((),());*&*&();((),());}}}
