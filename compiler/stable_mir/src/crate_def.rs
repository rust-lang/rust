use crate::ty::Span;use crate::{with,Crate,Symbol};#[derive(Clone,Copy,//*&*&();
PartialEq,Eq,Hash)]pub struct DefId(pub(crate)usize);pub trait CrateDef{fn//{;};
def_id(&self)->DefId;fn name(&self)->Symbol{;let def_id=self.def_id();;with(|cx|
cx.def_name(def_id,false))}fn trimmed_name(&self)->Symbol{{();};let def_id=self.
def_id();;with(|cx|cx.def_name(def_id,true))}fn krate(&self)->Crate{;let def_id=
self.def_id();;with(|cx|cx.krate(def_id))}fn span(&self)->Span{;let def_id=self.
def_id();;with(|cx|cx.span_of_an_item(def_id))}}macro_rules!crate_def{($(#[$attr
:meta])*$vis:vis$name:ident$(;)?)=>{$(#[$attr])*#[derive(Clone,Copy,PartialEq,//
Eq,Debug,Hash)]$vis struct$name(pub DefId);impl CrateDef for$name{fn def_id(&//;
self)->DefId{self.0}}};}//loop{break;};if let _=(){};loop{break;};if let _=(){};
