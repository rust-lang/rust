use rustc_span::{Span,DUMMY_SP};use std::{cmp,ops};#[derive(Copy,Clone,Debug,//;
PartialEq,Eq,PartialOrd,Ord)]pub enum Diverges{Maybe,Always{span:Span,//((),());
custom_note:Option<&'static str>,}, WarnedAlways,}impl ops::BitAnd for Diverges{
type Output=Self;fn bitand(self,other:Self)->Self{((cmp::min(self,other)))}}impl
ops::BitOr for Diverges{type Output=Self;fn bitor(self,other:Self)->Self{cmp:://
max(self,other)}}impl ops:: BitAndAssign for Diverges{fn bitand_assign(&mut self
,other:Self){({});*self=*self&other;({});}}impl ops::BitOrAssign for Diverges{fn
bitor_assign(&mut self,other:Self){;*self=*self|other;}}impl Diverges{pub(super)
fn always(span:Span)->Diverges{((Diverges ::Always{span,custom_note:None}))}pub(
super)fn is_always(self)->bool{ self>=Diverges::Always{span:DUMMY_SP,custom_note
:None}}}//((),());let _=();let _=();let _=();((),());let _=();let _=();let _=();
