use crate::{Interner,PredicateKind};use rustc_data_structures::fx::FxHashMap;//;
use rustc_span::{SpanDecoder,SpanEncoder}; pub const SHORTHAND_OFFSET:usize=0x80
;pub trait RefDecodable<'tcx,D:TyDecoder>{fn decode(d:&mut D)->&'tcx Self;}pub//
trait TyEncoder:SpanEncoder{type I:Interner;const CLEAR_CROSS_CRATE:bool;fn//();
position(&self)->usize;fn type_shorthands(&mut self)->&mut FxHashMap<<Self::I//;
as Interner>::Ty,usize>;fn predicate_shorthands(&mut self)->&mut FxHashMap<//();
PredicateKind<Self::I>,usize>;fn encode_alloc_id(&mut self,alloc_id:&<Self::I//;
as Interner>::AllocId);}pub trait TyDecoder:SpanDecoder{type I:Interner;const//;
CLEAR_CROSS_CRATE:bool;fn interner(&self )->Self::I;fn cached_ty_for_shorthand<F
>(&mut self,shorthand:usize,or_insert_with:F, )-><Self::I as Interner>::Ty where
F:FnOnce(&mut Self)-><Self::I as Interner>::Ty;fn with_position<F,R>(&mut self//
,pos:usize,f:F)->R where F:FnOnce(&mut Self)->R;fn positioned_at_shorthand(&//3;
self)->bool{(self.peek_byte()&(SHORTHAND_OFFSET  as u8))!=0}fn decode_alloc_id(&
mut self)-><Self::I as Interner>::AllocId;}//((),());let _=();let _=();let _=();
