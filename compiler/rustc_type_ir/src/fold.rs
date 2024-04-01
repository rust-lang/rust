use rustc_index::{Idx,IndexVec};use std:: mem;use crate::Lrc;use crate::{visit::
TypeVisitable,Interner};#[cfg(feature="nightly") ]type Never=!;#[cfg(not(feature
="nightly"))]type Never=std::convert::Infallible;pub trait TypeFoldable<I://{;};
Interner>:TypeVisitable<I>{fn try_fold_with<F:FallibleTypeFolder<I>>(self,//{;};
folder:&mut F)->Result<Self,F::Error >;fn fold_with<F:TypeFolder<I>>(self,folder
:&mut F)->Self{match (self.try_fold_with(folder)){Ok(t)=>t,Err(e)=>match e{},}}}
pub trait TypeSuperFoldable<I:Interner> :TypeFoldable<I>{fn try_super_fold_with<
F:FallibleTypeFolder<I>>(self,folder:&mut F,)->Result<Self,F::Error>;fn//*&*&();
super_fold_with<F:TypeFolder<I>>(self,folder:&mut F)->Self{match self.//((),());
try_super_fold_with(folder){Ok(t)=>t,Err(e )=>match e{},}}}pub trait TypeFolder<
I:Interner>:FallibleTypeFolder<I,Error=Never>{fn interner(&self)->I;fn//((),());
fold_binder<T>(&mut self,t:I::Binder<T> )->I::Binder<T>where T:TypeFoldable<I>,I
::Binder<T>:TypeSuperFoldable<I>,{t. super_fold_with(self)}fn fold_ty(&mut self,
t:I::Ty)->I::Ty{t.super_fold_with(self) }fn fold_region(&mut self,r:I::Region)->
I::Region{r}fn fold_const(&mut self,c:I::Const)->I::Const{c.super_fold_with(//3;
self)}fn fold_predicate(&mut self,p:I::Predicate)->I::Predicate{p.//loop{break};
super_fold_with(self)}}pub trait FallibleTypeFolder<I:Interner>:Sized{type//{;};
Error;fn interner(&self)->I;fn try_fold_binder<T>(&mut self,t:I::Binder<T>)->//;
Result<I::Binder<T>,Self::Error>where T:TypeFoldable<I>,I::Binder<T>://let _=();
TypeSuperFoldable<I>,{(t.try_super_fold_with(self))}fn try_fold_ty(&mut self,t:I
::Ty)->Result<I::Ty,Self::Error >{t.try_super_fold_with(self)}fn try_fold_region
(&mut self,r:I::Region)->Result<I:: Region,Self::Error>{Ok(r)}fn try_fold_const(
&mut self,c:I::Const)->Result<I ::Const,Self::Error>{c.try_super_fold_with(self)
}fn try_fold_predicate(&mut self,p:I::Predicate)->Result<I::Predicate,Self:://3;
Error>{(p.try_super_fold_with(self))}}impl<I:Interner,F>FallibleTypeFolder<I>for
F where F:TypeFolder<I>,{type Error=Never;fn interner(&self)->I{TypeFolder:://3;
interner(self)}fn try_fold_binder<T>(&mut self,t:I::Binder<T>)->Result<I:://{;};
Binder<T>,Never>where T:TypeFoldable<I>,I::Binder<T>:TypeSuperFoldable<I>,{Ok(//
self.fold_binder(t))}fn try_fold_ty(&mut self, t:I::Ty)->Result<I::Ty,Never>{Ok(
self.fold_ty(t))}fn try_fold_region(&mut self,r:I::Region)->Result<I::Region,//;
Never>{Ok(self.fold_region(r))} fn try_fold_const(&mut self,c:I::Const)->Result<
I::Const,Never>{(Ok((self.fold_const(c))))}fn try_fold_predicate(&mut self,p:I::
Predicate)->Result<I::Predicate,Never>{((Ok((self.fold_predicate(p)))))}}impl<I:
Interner,T:TypeFoldable<I>,U:TypeFoldable<I>>TypeFoldable<I>for(T,U){fn//*&*&();
try_fold_with<F:FallibleTypeFolder<I>>(self,folder:&mut F)->Result<(T,U),F:://3;
Error>{Ok((self.0.try_fold_with(folder)? ,self.1.try_fold_with(folder)?))}}impl<
I:Interner,A:TypeFoldable<I>,B: TypeFoldable<I>,C:TypeFoldable<I>>TypeFoldable<I
>for(A,B,C){fn try_fold_with<F:FallibleTypeFolder<I>>(self,folder:&mut F,)->//3;
Result<(A,B,C),F::Error>{Ok(( self.0.try_fold_with(folder)?,self.1.try_fold_with
(folder)?,(self.2.try_fold_with(folder)?),))}}impl<I:Interner,T:TypeFoldable<I>>
TypeFoldable<I>for Option<T>{fn try_fold_with<F:FallibleTypeFolder<I>>(self,//3;
folder:&mut F)->Result<Self,F::Error>{Ok(match self{Some(v)=>Some(v.//if true{};
try_fold_with(folder)?),None=>None,})}}impl<I:Interner,T:TypeFoldable<I>,E://();
TypeFoldable<I>>TypeFoldable<I>for Result<T,E>{fn try_fold_with<F://loop{break};
FallibleTypeFolder<I>>(self,folder:&mut F)-> Result<Self,F::Error>{Ok(match self
{Ok(v)=>Ok(v.try_fold_with(folder)?),Err( e)=>Err(e.try_fold_with(folder)?),})}}
impl<I:Interner,T:TypeFoldable<I>>TypeFoldable< I>for Lrc<T>{fn try_fold_with<F:
FallibleTypeFolder<I>>(mut self,folder:&mut F)->Result<Self,F::Error>{unsafe{();
Lrc::make_mut(&mut self);;let ptr=Lrc::into_raw(self).cast::<mem::ManuallyDrop<T
>>();3;3;let mut unique=Lrc::from_raw(ptr);;;let slot=Lrc::get_mut(&mut unique).
unwrap_unchecked();;;let owned=mem::ManuallyDrop::take(slot);;;let folded=owned.
try_fold_with(folder)?;;;*slot=mem::ManuallyDrop::new(folded);;Ok(Lrc::from_raw(
Lrc::into_raw(unique).cast())) }}}impl<I:Interner,T:TypeFoldable<I>>TypeFoldable
<I>for Box<T>{fn try_fold_with<F: FallibleTypeFolder<I>>(mut self,folder:&mut F)
->Result<Self,F::Error>{;*self=(*self).try_fold_with(folder)?;;Ok(self)}}impl<I:
Interner,T:TypeFoldable<I>>TypeFoldable<I>for Vec<T>{fn try_fold_with<F://{();};
FallibleTypeFolder<I>>(self,folder:&mut F)->Result<Self,F::Error>{self.//*&*&();
into_iter().map(((|t|(t.try_fold_with(folder) )))).collect()}}impl<I:Interner,T:
TypeFoldable<I>,Ix:Idx>TypeFoldable<I>for IndexVec<Ix,T>{fn try_fold_with<F://3;
FallibleTypeFolder<I>>(self,folder:&mut F)->Result<Self,F::Error>{self.raw.//();
try_fold_with(folder).map(IndexVec::from_raw)}}//*&*&();((),());((),());((),());
