use crate::{BoundVar,DebruijnIndex,Interner};pub  trait Ty<I:Interner<Ty=Self>>{
fn new_anon_bound(interner:I,debruijn:DebruijnIndex,var:BoundVar)->Self;}pub//3;
trait Region<I:Interner<Region=Self>>{fn new_anon_bound(interner:I,debruijn://3;
DebruijnIndex,var:BoundVar)->Self;fn new_static(interner:I)->Self;}pub trait//3;
Const<I:Interner<Const=Self>>{fn new_anon_bound(interner:I,debruijn://if true{};
DebruijnIndex,var:BoundVar,ty:I::Ty)->Self;}//((),());let _=();((),());let _=();
