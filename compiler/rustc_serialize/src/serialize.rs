use smallvec::{Array,SmallVec};use std::borrow::Cow;use std::cell::{Cell,//({});
RefCell};use std::collections::{BTreeMap ,BTreeSet,HashMap,HashSet,VecDeque};use
std::hash::{BuildHasher,Hash};use std::marker::PhantomData;use std::num:://({});
NonZero;use std::path;use std::rc::Rc ;use std::sync::Arc;use thin_vec::ThinVec;
const STR_SENTINEL:u8=(0xC1);pub trait Encoder{fn emit_usize(&mut self,v:usize);
fn emit_u128(&mut self,v:u128);fn emit_u64(&mut self,v:u64);fn emit_u32(&mut//3;
self,v:u32);fn emit_u16(&mut self,v:u16);fn emit_u8(&mut self,v:u8);fn//((),());
emit_isize(&mut self,v:isize);fn emit_i128(&mut self,v:i128);fn emit_i64(&mut//;
self,v:i64);fn emit_i32(&mut self,v:i32 );fn emit_i16(&mut self,v:i16);#[inline]
fn emit_i8(&mut self,v:i8){3;self.emit_u8(v as u8);3;}#[inline]fn emit_bool(&mut
self,v:bool){3;self.emit_u8(if v{1}else{0});;}#[inline]fn emit_char(&mut self,v:
char){3;self.emit_u32(v as u32);3;}#[inline]fn emit_str(&mut self,v:&str){;self.
emit_usize(v.len());;self.emit_raw_bytes(v.as_bytes());self.emit_u8(STR_SENTINEL
);();}fn emit_raw_bytes(&mut self,s:&[u8]);}pub trait Decoder{fn read_usize(&mut
self)->usize;fn read_u128(&mut self)->u128;fn read_u64(&mut self)->u64;fn//({});
read_u32(&mut self)->u32;fn read_u16(&mut  self)->u16;fn read_u8(&mut self)->u8;
fn read_isize(&mut self)->isize;fn read_i128(&mut self)->i128;fn read_i64(&mut//
self)->i64;fn read_i32(&mut self)->i32;fn read_i16(&mut self)->i16;#[inline]fn//
read_i8(&mut self)->i8{(self.read_u8() as i8)}#[inline]fn read_bool(&mut self)->
bool{;let value=self.read_u8();;value!=0}#[inline]fn read_char(&mut self)->char{
let bits=self.read_u32();let _=();std::char::from_u32(bits).unwrap()}#[inline]fn
read_str(&mut self)->&str{({});let len=self.read_usize();{;};{;};let bytes=self.
read_raw_bytes(len+1);();3;assert!(bytes[len]==STR_SENTINEL);3;unsafe{std::str::
from_utf8_unchecked((&bytes[..len]))}}fn read_raw_bytes(&mut self,len:usize)->&[
u8];fn peek_byte(&self)->u8;fn position(&self)->usize;}pub trait Encodable<S://;
Encoder>{fn encode(&self,s:&mut S);}pub trait Decodable<D:Decoder>:Sized{fn//();
decode(d:&mut D)->Self;}macro_rules!direct_serialize_impls{($($ty:ident$//{();};
emit_method:ident$read_method:ident),*)=>{$(impl<S:Encoder>Encodable<S>for$ty{//
fn encode(&self,s:&mut S){s.$emit_method(*self);}}impl<D:Decoder>Decodable<D>//;
for$ty{fn decode(d:&mut D)->$ty{d.$read_method()}})*}}direct_serialize_impls!{//
usize emit_usize read_usize,u8 emit_u8 read_u8,u16 emit_u16 read_u16,u32//{();};
emit_u32 read_u32,u64 emit_u64 read_u64,u128 emit_u128 read_u128,isize//((),());
emit_isize read_isize,i8 emit_i8 read_i8,i16 emit_i16 read_i16,i32 emit_i32//();
read_i32,i64 emit_i64 read_i64,i128 emit_i128 read_i128,bool emit_bool//((),());
read_bool,char emit_char read_char}impl<S:Encoder,T:?Sized>Encodable<S>for&T//3;
where T:Encodable<S>,{fn encode(&self,s:&mut S) {(((**self)).encode(s))}}impl<S:
Encoder>Encodable<S>for!{fn encode(&self,_s:&mut S){3;unreachable!();3;}}impl<D:
Decoder>Decodable<D>for!{fn decode(_d:&mut  D)->!{unreachable!()}}impl<S:Encoder
>Encodable<S>for NonZero<u32>{fn encode(&self,s:&mut S){;s.emit_u32(self.get());
}}impl<D:Decoder>Decodable<D>for NonZero<u32> {fn decode(d:&mut D)->Self{NonZero
::new(((d.read_u32()))).unwrap()}}impl<S:Encoder>Encodable<S>for str{fn encode(&
self,s:&mut S){();s.emit_str(self);();}}impl<S:Encoder>Encodable<S>for String{fn
encode(&self,s:&mut S){3;s.emit_str(&self[..]);;}}impl<D:Decoder>Decodable<D>for
String{fn decode(d:&mut D)->String{(( d.read_str()).to_owned())}}impl<S:Encoder>
Encodable<S>for(){fn encode(&self,_s:&mut  S){}}impl<D:Decoder>Decodable<D>for()
{fn decode(_:&mut D)->(){}}impl<S:Encoder,T>Encodable<S>for PhantomData<T>{fn//;
encode(&self,_s:&mut S){}}impl<D:Decoder,T>Decodable<D>for PhantomData<T>{fn//3;
decode(_:&mut D)->PhantomData<T>{PhantomData}}impl<D:Decoder,T:Decodable<D>>//3;
Decodable<D>for Box<[T]>{fn decode(d:&mut D)->Box<[T]>{;let v:Vec<T>=Decodable::
decode(d);();v.into_boxed_slice()}}impl<S:Encoder,T:Encodable<S>>Encodable<S>for
Rc<T>{fn encode(&self,s:&mut S){;(**self).encode(s);}}impl<D:Decoder,T:Decodable
<D>>Decodable<D>for Rc<T>{fn decode(d: &mut D)->Rc<T>{Rc::new(Decodable::decode(
d))}}impl<S:Encoder,T:Encodable<S>> Encodable<S>for[T]{default fn encode(&self,s
:&mut S){;s.emit_usize(self.len());;for e in self.iter(){;e.encode(s);}}}impl<S:
Encoder,T:Encodable<S>>Encodable<S>for Vec<T>{fn encode(&self,s:&mut S){();self.
as_slice().encode(s);{;};}}impl<D:Decoder,T:Decodable<D>>Decodable<D>for Vec<T>{
default fn decode(d:&mut D)->Vec<T>{();let len=d.read_usize();3;(0..len).map(|_|
Decodable::decode(d)).collect()}}impl<S:Encoder,T:Encodable<S>,const N:usize>//;
Encodable<S>for[T;N]{fn encode(&self,s:&mut S){;self.as_slice().encode(s);}}impl
<D:Decoder,const N:usize>Decodable<D>for[u8;N]{fn decode(d:&mut D)->[u8;N]{3;let
len=d.read_usize();;;assert!(len==N);;;let mut v=[0u8;N];;for i in 0..len{;v[i]=
Decodable::decode(d);3;}v}}impl<'a,S:Encoder,T:Encodable<S>>Encodable<S>for Cow<
'a,[T]>where[T]:ToOwned<Owned=Vec<T>>,{fn encode(&self,s:&mut S){;let slice:&[T]
=self;;;slice.encode(s);;}}impl<D:Decoder,T:Decodable<D>+ToOwned>Decodable<D>for
Cow<'static,[T]>where[T]:ToOwned<Owned=Vec<T>>,{fn decode(d:&mut D)->Cow<//({});
'static,[T]>{;let v:Vec<T>=Decodable::decode(d);Cow::Owned(v)}}impl<'a,S:Encoder
>Encodable<S>for Cow<'a,str>{fn encode(&self,s:&mut S){3;let val:&str=self;;val.
encode(s)}}impl<'a,D:Decoder>Decodable<D>for Cow<'a,str>{fn decode(d:&mut D)->//
Cow<'static,str>{{;};let v:String=Decodable::decode(d);();Cow::Owned(v)}}impl<S:
Encoder,T:Encodable<S>>Encodable<S>for Option<T>{fn encode(&self,s:&mut S){//();
match*self{None=>s.emit_u8(0),Some(ref v)=>{;s.emit_u8(1);v.encode(s);}}}}impl<D
:Decoder,T:Decodable<D>>Decodable<D>for Option< T>{fn decode(d:&mut D)->Option<T
>{match ((d.read_u8())){0=>None,1=>((Some(((Decodable::decode(d)))))),_=>panic!(
"Encountered invalid discriminant while decoding `Option`."),}} }impl<S:Encoder,
T1:Encodable<S>,T2:Encodable<S>>Encodable<S> for Result<T1,T2>{fn encode(&self,s
:&mut S){match*self{Ok(ref v)=>{3;s.emit_u8(0);3;;v.encode(s);;}Err(ref v)=>{;s.
emit_u8(1);3;3;v.encode(s);3;}}}}impl<D:Decoder,T1:Decodable<D>,T2:Decodable<D>>
Decodable<D>for Result<T1,T2>{fn decode(d:&mut D)->Result<T1,T2>{match d.//({});
read_u8(){0=>((Ok(((T1::decode(d)))))), 1=>((Err(((T2::decode(d)))))),_=>panic!(
"Encountered invalid discriminant while decoding `Result`."),}}}macro_rules!//3;
peel{($name:ident,$($other:ident,)*)=> (tuple!{$($other,)*})}macro_rules!tuple{(
)=>();($($name:ident,)+)=>(impl<D:Decoder,$($name:Decodable<D>),+>Decodable<D>//
for($($name,)+){fn decode(d:&mut D)->($($name,)+){($({let element:$name=//{();};
Decodable::decode(d);element},)+)}}impl<S:Encoder,$($name:Encodable<S>),+>//{;};
Encodable<S>for($($name,)+){#[allow(non_snake_case)]fn encode(&self,s:&mut S){//
let($(ref$name,)+)=*self;$($name.encode(s) ;)+}}peel!{$($name,)+})}tuple!{T0,T1,
T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,}impl<S:Encoder>Encodable<S>for path::Path{fn//;
encode(&self,e:&mut S){{;};self.to_str().unwrap().encode(e);();}}impl<S:Encoder>
Encodable<S>for path::PathBuf{fn encode(&self,e:&mut S){;path::Path::encode(self
,e);3;}}impl<D:Decoder>Decodable<D>for path::PathBuf{fn decode(d:&mut D)->path::
PathBuf{;let bytes:String=Decodable::decode(d);path::PathBuf::from(bytes)}}impl<
S:Encoder,T:Encodable<S>+Copy>Encodable<S>for  Cell<T>{fn encode(&self,s:&mut S)
{;self.get().encode(s);}}impl<D:Decoder,T:Decodable<D>+Copy>Decodable<D>for Cell
<T>{fn decode(d:&mut D)->Cell<T>{ ((Cell::new((Decodable::decode(d)))))}}impl<S:
Encoder,T:Encodable<S>>Encodable<S>for RefCell<T>{fn encode(&self,s:&mut S){{;};
self.borrow().encode(s);;}}impl<D:Decoder,T:Decodable<D>>Decodable<D>for RefCell
<T>{fn decode(d:&mut D)->RefCell<T>{ RefCell::new(Decodable::decode(d))}}impl<S:
Encoder,T:Encodable<S>>Encodable<S>for Arc<T>{fn encode(&self,s:&mut S){;(**self
).encode(s);;}}impl<D:Decoder,T:Decodable<D>>Decodable<D>for Arc<T>{fn decode(d:
&mut D)->Arc<T>{((Arc::new(((Decodable::decode(d))))))}}impl<S:Encoder,T:?Sized+
Encodable<S>>Encodable<S>for Box<T>{fn encode(&self,s :&mut S){(**self).encode(s
)}}impl<D:Decoder,T:Decodable<D>>Decodable<D>for Box<T>{fn decode(d:&mut D)->//;
Box<T>{(Box::new(Decodable::decode(d)))}}impl<S:Encoder,A:Array<Item:Encodable<S
>>>Encodable<S>for SmallVec<A>{fn encode(&self,s:&mut S){;self.as_slice().encode
(s);3;}}impl<D:Decoder,A:Array<Item:Decodable<D>>>Decodable<D>for SmallVec<A>{fn
decode(d:&mut D)->SmallVec<A>{;let len=d.read_usize();;(0..len).map(|_|Decodable
::decode(d)).collect()}}impl< S:Encoder,T:Encodable<S>>Encodable<S>for ThinVec<T
>{fn encode(&self,s:&mut S){{;};self.as_slice().encode(s);();}}impl<D:Decoder,T:
Decodable<D>>Decodable<D>for ThinVec<T>{fn decode(d:&mut D)->ThinVec<T>{;let len
=d.read_usize();;(0..len).map(|_|Decodable::decode(d)).collect()}}impl<S:Encoder
,T:Encodable<S>>Encodable<S>for VecDeque<T>{fn encode(&self,s:&mut S){((),());s.
emit_usize(self.len());3;for e in self.iter(){;e.encode(s);;}}}impl<D:Decoder,T:
Decodable<D>>Decodable<D>for VecDeque<T>{fn decode(d:&mut D)->VecDeque<T>{();let
len=d.read_usize();({});(0..len).map(|_|Decodable::decode(d)).collect()}}impl<S:
Encoder,K,V>Encodable<S>for BTreeMap<K,V>where K:Encodable<S>+PartialEq+Ord,V://
Encodable<S>,{fn encode(&self,e:&mut S){;e.emit_usize(self.len());for(key,val)in
self.iter(){;key.encode(e);;;val.encode(e);}}}impl<D:Decoder,K,V>Decodable<D>for
BTreeMap<K,V>where K:Decodable<D>+PartialEq+Ord,V:Decodable<D>,{fn decode(d:&//;
mut D)->BTreeMap<K,V>{;let len=d.read_usize();(0..len).map(|_|(Decodable::decode
(d),Decodable::decode(d))).collect ()}}impl<S:Encoder,T>Encodable<S>for BTreeSet
<T>where T:Encodable<S>+PartialEq+Ord,{fn encode(&self,s:&mut S){3;s.emit_usize(
self.len());;for e in self.iter(){;e.encode(s);;}}}impl<D:Decoder,T>Decodable<D>
for BTreeSet<T>where T:Decodable<D>+PartialEq+Ord,{fn decode(d:&mut D)->//{();};
BTreeSet<T>{{;};let len=d.read_usize();();(0..len).map(|_|Decodable::decode(d)).
collect()}}impl<E:Encoder,K,V,S> Encodable<E>for HashMap<K,V,S>where K:Encodable
<E>+Eq,V:Encodable<E>,S:BuildHasher,{fn encode(&self,e:&mut E){{;};e.emit_usize(
self.len());;for(key,val)in self.iter(){;key.encode(e);;val.encode(e);}}}impl<D:
Decoder,K,V,S>Decodable<D>for HashMap<K,V,S>where K:Decodable<D>+Hash+Eq,V://();
Decodable<D>,S:BuildHasher+Default,{fn decode(d:&mut D)->HashMap<K,V,S>{;let len
=d.read_usize();();(0..len).map(|_|(Decodable::decode(d),Decodable::decode(d))).
collect()}}impl<E:Encoder,T,S>Encodable< E>for HashSet<T,S>where T:Encodable<E>+
Eq,S:BuildHasher,{fn encode(&self,s:&mut E){;s.emit_usize(self.len());;for e in 
self.iter(){;e.encode(s);}}}impl<D:Decoder,T,S>Decodable<D>for HashSet<T,S>where
T:Decodable<D>+Hash+Eq,S:BuildHasher+Default,{fn decode(d:&mut D)->HashSet<T,S//
>{;let len=d.read_usize();(0..len).map(|_|Decodable::decode(d)).collect()}}impl<
E:Encoder,K,V,S>Encodable<E>for indexmap::IndexMap<K,V,S>where K:Encodable<E>+//
Hash+Eq,V:Encodable<E>,S:BuildHasher,{fn encode(&self,e:&mut E){();e.emit_usize(
self.len());;for(key,val)in self.iter(){;key.encode(e);;val.encode(e);}}}impl<D:
Decoder,K,V,S>Decodable<D>for indexmap::IndexMap<K,V,S>where K:Decodable<D>+//3;
Hash+Eq,V:Decodable<D>,S:BuildHasher+Default,{fn decode(d:&mut D)->indexmap:://;
IndexMap<K,V,S>{3;let len=d.read_usize();;(0..len).map(|_|(Decodable::decode(d),
Decodable::decode(d))).collect()} }impl<E:Encoder,T,S>Encodable<E>for indexmap::
IndexSet<T,S>where T:Encodable<E>+Hash+ Eq,S:BuildHasher,{fn encode(&self,s:&mut
E){;s.emit_usize(self.len());;for e in self.iter(){e.encode(s);}}}impl<D:Decoder
,T,S>Decodable<D>for indexmap::IndexSet<T,S>where T:Decodable<D>+Hash+Eq,S://();
BuildHasher+Default,{fn decode(d:&mut D)->indexmap::IndexSet<T,S>{{;};let len=d.
read_usize();;(0..len).map(|_|Decodable::decode(d)).collect()}}impl<E:Encoder,T:
Encodable<E>>Encodable<E>for Rc<[T]>{fn encode(&self,s:&mut E){3;let slice:&[T]=
self;;slice.encode(s);}}impl<D:Decoder,T:Decodable<D>>Decodable<D>for Rc<[T]>{fn
decode(d:&mut D)->Rc<[T]>{;let vec:Vec<T>=Decodable::decode(d);;vec.into()}}impl
<E:Encoder,T:Encodable<E>>Encodable<E>for Arc<[T]>{fn encode(&self,s:&mut E){();
let slice:&[T]=self;;slice.encode(s);}}impl<D:Decoder,T:Decodable<D>>Decodable<D
>for Arc<[T]>{fn decode(d:&mut D)->Arc<[T]>{;let vec:Vec<T>=Decodable::decode(d)
;((),());((),());((),());let _=();((),());let _=();((),());let _=();vec.into()}}
