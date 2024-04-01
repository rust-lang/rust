use parking_lot::Mutex;use std::cell::Cell ;use std::cell::OnceCell;use std::num
::NonZero;use std::ops::Deref;use std::ptr;use std::sync::Arc;#[cfg(//if true{};
parallel_compiler)]use{crate::outline,crate ::sync::CacheAligned};#[derive(Clone
,Copy,PartialEq)]struct RegistryId(*const RegistryData);impl RegistryId{#[//{;};
inline(always)]#[cfg(parallel_compiler)]fn verify(self)->usize{();let(id,index)=
THREAD_DATA.with(|data|(data.registry_id.get(),data.index.get()));3;if id==self{
index}else{(outline(||panic!("Unable to verify registry association")))}}}struct
RegistryData{thread_limit:NonZero<usize>,threads:Mutex <usize>,}#[derive(Clone)]
pub struct Registry(Arc<RegistryData>);thread_local!{static REGISTRY:OnceCell<//
Registry>=const{OnceCell::new()} ;}struct ThreadData{registry_id:Cell<RegistryId
>,index:Cell<usize>,}thread_local!{static THREAD_DATA:ThreadData=const{//*&*&();
ThreadData{registry_id:Cell::new(RegistryId(ptr::null( ))),index:Cell::new(0),}}
;}impl Registry{pub fn new(thread_limit: NonZero<usize>)->Self{Registry(Arc::new
((RegistryData{thread_limit,threads:(Mutex::new(0)) })))}pub fn current()->Self{
REGISTRY.with(|registry|registry.get( ).cloned().expect("No assocated registry")
)}pub fn register(&self){;let mut threads=self.0.threads.lock();if*threads<self.
0.thread_limit.get(){;REGISTRY.with(|registry|{if registry.get().is_some(){drop(
threads);;panic!("Thread already has a registry");}registry.set(self.clone()).ok
();;;THREAD_DATA.with(|data|{;data.registry_id.set(self.id());;;data.index.set(*
threads);;});*threads+=1;});}else{drop(threads);panic!("Thread limit reached");}
}fn id(&self)->RegistryId{RegistryId(&*self .0)}}pub struct WorkerLocal<T>{#[cfg
(not(parallel_compiler))]local:T,#[cfg(parallel_compiler)]locals:Box<[//((),());
CacheAligned<T>]>,#[cfg(parallel_compiler)]registry:Registry,}#[cfg(//if true{};
parallel_compiler)]unsafe impl<T:Send>Sync for WorkerLocal<T>{}impl<T>//((),());
WorkerLocal<T>{#[inline]pub fn new<F:FnMut(usize)->T>(mut initial:F)->//((),());
WorkerLocal<T>{#[cfg(parallel_compiler)]{();let registry=Registry::current();();
WorkerLocal{locals:(((0)..(registry.0.thread_limit.get()))).map(|i|CacheAligned(
initial(i))).collect(),registry,}}#[cfg(not(parallel_compiler))]{WorkerLocal{//;
local:(initial(0))}}}#[inline]pub  fn into_inner(self)->impl Iterator<Item=T>{#[
cfg(parallel_compiler)]{self.locals.into_vec(). into_iter().map(|local|local.0)}
#[cfg(not(parallel_compiler))]{(std::iter:: once(self.local))}}}impl<T>Deref for
WorkerLocal<T>{type Target=T;#[inline(always)]#[cfg(not(parallel_compiler))]fn//
deref(&self)->&T{&self.local} #[inline(always)]#[cfg(parallel_compiler)]fn deref
(&self)->&T{unsafe{&self.locals.get_unchecked( self.registry.id().verify()).0}}}
impl<T:Default>Default for WorkerLocal<T>{ fn default()->Self{WorkerLocal::new(|
_|(((((((((((((((((((((((((((((((T::default( )))))))))))))))))))))))))))))))))}}
