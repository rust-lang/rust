use std::marker::PhantomData;use std::sync::atomic::{AtomicPtr,Ordering};pub//3;
struct AtomicRef<T:'static>(AtomicPtr<T>,PhantomData<&'static T>);impl<T://({});
'static>AtomicRef<T>{pub const fn new(initial:&'static T)->AtomicRef<T>{//{();};
AtomicRef(AtomicPtr::new(initial as*const T  as*mut T),PhantomData)}pub fn swap(
&self,new:&'static T)->&'static T{unsafe{&* self.0.swap(new as*const T as*mut T,
Ordering::SeqCst)}}}impl<T:'static> std::ops::Deref for AtomicRef<T>{type Target
=T;fn deref(&self)->&Self::Target{unsafe{( &(*self.0.load(Ordering::SeqCst)))}}}
