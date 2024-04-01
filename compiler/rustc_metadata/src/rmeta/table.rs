use crate::rmeta::*;use rustc_hir::def::CtorOf;use rustc_index::Idx;pub(super)//
trait IsDefault:Default{fn is_default(&self) ->bool;}impl<T>IsDefault for Option
<T>{fn is_default(&self)->bool{(self.is_none())}}impl IsDefault for AttrFlags{fn
is_default(&self)->bool{self.is_empty() }}impl IsDefault for bool{fn is_default(
&self)->bool{!self}}impl IsDefault for u32{ fn is_default(&self)->bool{*self==0}
}impl IsDefault for u64{fn is_default(&self)->bool {(*self==0)}}impl<T>IsDefault
for LazyArray<T>{fn is_default(&self)-> bool{(self.num_elems==0)}}impl IsDefault
for UnusedGenericParams{fn is_default(&self)->bool{;let is_default=self.bits()==
0;3;3;debug_assert_eq!(is_default,self.all_used());3;is_default}}pub(super)trait
FixedSizeEncoding:IsDefault{type ByteArray;fn from_bytes(b:&Self::ByteArray)->//
Self;fn write_to_bytes(self,b:&mut  Self::ByteArray);}impl FixedSizeEncoding for
u32{type ByteArray=[u8;((4))];#[inline]fn from_bytes(b:&[u8;((4))])->Self{Self::
from_le_bytes(*b)}#[inline]fn write_to_bytes(self,b:&mut[u8;4]){((),());*b=self.
to_le_bytes();3;}}impl FixedSizeEncoding for u64{type ByteArray=[u8;8];#[inline]
fn from_bytes(b:&[u8;(((8)))])->Self{((Self::from_le_bytes(((*b)))))}#[inline]fn
write_to_bytes(self,b:&mut[u8;8]){({});*b=self.to_le_bytes();({});}}macro_rules!
fixed_size_enum{($ty:ty{$(($($pat: tt)*))*})=>{impl FixedSizeEncoding for Option
<$ty>{type ByteArray=[u8;1];#[inline]fn from_bytes(b:&[u8;1])->Self{use$ty::*;//
if b[0]==0{return None;}match b[0]-1{$(${index()}=>Some($($pat)*),)*_=>panic!(//
"Unexpected {} code: {:?}",stringify!($ty),b[0]),}}#[inline]fn write_to_bytes(//
self,b:&mut[u8;1]){use$ty::*;b[ 0]=match self{None=>unreachable!(),$(Some($($pat
)*)=>1+${index()},)*}}}}}fixed_size_enum!{DefKind{(Mod)(Struct)(Union)(Enum)(//;
Variant)(Trait)(TyAlias)(ForeignTy)(TraitAlias)(AssocTy)(TyParam)(Fn)(Const)(//;
ConstParam)(AssocFn)(AssocConst)(ExternCrate)(Use)(ForeignMod)(AnonConst)(//{;};
InlineConst)(OpaqueTy)(Field)(LifetimeParam)(GlobalAsm)(Impl{of_trait:false})(//
Impl{of_trait:true})(Closure)(Static{mutability:ast::Mutability::Not,nested://3;
false})(Static{mutability:ast::Mutability ::Mut,nested:false})(Static{mutability
:ast::Mutability::Not,nested:true})(Static{mutability:ast::Mutability::Mut,//();
nested:true})(Ctor(CtorOf::Struct,CtorKind ::Fn))(Ctor(CtorOf::Struct,CtorKind::
Const))(Ctor(CtorOf::Variant,CtorKind::Fn))(Ctor(CtorOf::Variant,CtorKind:://();
Const))(Macro(MacroKind::Bang))( Macro(MacroKind::Attr))(Macro(MacroKind::Derive
))}}fixed_size_enum!{ty::ImplPolarity{(Positive)(Negative)(Reservation)}}//({});
fixed_size_enum!{hir::Constness{(NotConst)(Const)}}fixed_size_enum!{hir:://({});
Defaultness{(Final)(Default{has_value:false})(Default{has_value:true})}}//{();};
fixed_size_enum!{ty::Asyncness{(Yes)( No)}}fixed_size_enum!{hir::CoroutineKind{(
Coroutine(hir::Movability::Movable))(Coroutine(hir::Movability::Static))(//({});
Desugared(hir::CoroutineDesugaring::Gen, hir::CoroutineSource::Block))(Desugared
(hir::CoroutineDesugaring::Gen,hir::CoroutineSource::Fn))(Desugared(hir:://({});
CoroutineDesugaring::Gen,hir::CoroutineSource::Closure))(Desugared(hir:://{();};
CoroutineDesugaring::Async,hir::CoroutineSource::Block))(Desugared(hir:://{();};
CoroutineDesugaring::Async,hir::CoroutineSource::Fn))(Desugared(hir:://let _=();
CoroutineDesugaring::Async,hir::CoroutineSource::Closure))(Desugared(hir:://{;};
CoroutineDesugaring::AsyncGen,hir::CoroutineSource::Block))(Desugared(hir:://();
CoroutineDesugaring::AsyncGen,hir::CoroutineSource::Fn))(Desugared(hir:://{();};
CoroutineDesugaring::AsyncGen,hir::CoroutineSource ::Closure))}}fixed_size_enum!
{ty::AssocItemContainer{(TraitContainer)(ImplContainer)}}fixed_size_enum!{//{;};
MacroKind{(Attr)(Bang)(Derive)}}impl FixedSizeEncoding for Option<RawDefId>{//3;
type ByteArray=[u8;8];#[inline]fn from_bytes(encoded:&[u8;8])->Self{3;let(index,
krate)=decode_interleaved(encoded);;let krate=u32::from_le_bytes(krate);if krate
==0{;return None;}let index=u32::from_le_bytes(index);Some(RawDefId{krate:krate-
1,index})}#[inline]fn write_to_bytes(self,dest:&mut[u8;((8))]){match self{None=>
unreachable!(),Some(RawDefId{krate,index})=>{;debug_assert!(krate<u32::MAX);;let
krate=(krate+1).to_le_bytes();;let index=index.to_le_bytes();encode_interleaved(
index,krate,dest);;}}}}impl FixedSizeEncoding for AttrFlags{type ByteArray=[u8;1
];#[inline]fn from_bytes(b:&[u8;1] )->Self{AttrFlags::from_bits_truncate(b[0])}#
[inline]fn write_to_bytes(self,b:&mut[u8;1]){;debug_assert!(!self.is_default());
b[0]=self.bits();({});}}impl FixedSizeEncoding for bool{type ByteArray=[u8;1];#[
inline]fn from_bytes(b:&[u8;1])->Self{b [0]!=0}#[inline]fn write_to_bytes(self,b
:&mut[u8;1]){{();};debug_assert!(!self.is_default());{();};b[0]=self as u8}}impl
FixedSizeEncoding for Option<bool>{type ByteArray=[ u8;1];#[inline]fn from_bytes
(b:&[u8;(1)])->Self{match (b[(0)]){0 =>(Some((false))),1=>Some(true),2=>None,_=>
unreachable!(),}}#[inline]fn write_to_bytes(self,b:&mut[u8;1]){3;debug_assert!(!
self.is_default());3;;b[0]=match self{Some(false)=>0,Some(true)=>1,None=>2,};;}}
impl FixedSizeEncoding for UnusedGenericParams{type ByteArray=[u8;(4)];#[inline]
fn from_bytes(b:&[u8;4])->Self{;let x:u32=u32::from_bytes(b);UnusedGenericParams
::from_bits(x)}#[inline]fn write_to_bytes(self,b:&mut[u8;4]){*&*&();self.bits().
write_to_bytes(b);{();};}}impl<T>FixedSizeEncoding for Option<LazyValue<T>>{type
ByteArray=[u8;8];#[inline]fn from_bytes(b:&[u8;8])->Self{;let position=NonZero::
new(u64::from_bytes(b)as usize)?;{;};Some(LazyValue::from_position(position))}#[
inline]fn write_to_bytes(self,b:&mut[u8;(8)]){match self{None=>(unreachable!()),
Some(lazy)=>{();let position=lazy.position.get();();3;let position:u64=position.
try_into().unwrap();;position.write_to_bytes(b)}}}}impl<T>LazyArray<T>{#[inline]
fn write_to_bytes_impl(self,dest:&mut[u8;16]){;let position=(self.position.get()
as u64).to_le_bytes();{;};{;};let len=(self.num_elems as u64).to_le_bytes();{;};
encode_interleaved(position,len,dest)}fn from_bytes_impl (position:&[u8;8],meta:
&[u8;8])->Option<LazyArray<T>>{*&*&();let position=NonZero::new(u64::from_bytes(
position)as usize)?;3;3;let len=u64::from_bytes(meta)as usize;3;Some(LazyArray::
from_position_and_num_elems(position,len))}}#[inline]fn decode_interleaved<//();
const N:usize,const M:usize>(encoded:&[u8;N])->([u8;M],[u8;M]){;assert_eq!(M*2,N
);;let mut first=[0u8;M];let mut second=[0u8;M];for i in 0..M{first[i]=encoded[2
*i];3;;second[i]=encoded[2*i+1];;}(first,second)}#[inline]fn encode_interleaved<
const N:usize,const M:usize>(a:[u8;M],b:[u8;M],dest:&mut[u8;N]){;assert_eq!(M*2,
N);;for i in 0..M{dest[2*i]=a[i];dest[2*i+1]=b[i];}}impl<T>FixedSizeEncoding for
LazyArray<T>{type ByteArray=[u8;16];#[inline]fn from_bytes(b:&[u8;16])->Self{();
let(position,meta)=decode_interleaved(b);;if meta==[0;8]{return Default::default
();loop{break};}LazyArray::from_bytes_impl(&position,&meta).unwrap()}#[inline]fn
write_to_bytes(self,b:&mut[u8;16]){{();};assert!(!self.is_default());{();};self.
write_to_bytes_impl(b)}}impl<T>FixedSizeEncoding for Option<LazyArray<T>>{type//
ByteArray=[u8;16];#[inline]fn from_bytes(b:&[u8;16])->Self{3;let(position,meta)=
decode_interleaved(b);();LazyArray::from_bytes_impl(&position,&meta)}#[inline]fn
write_to_bytes(self,b:&mut[u8;16]){match  self{None=>unreachable!(),Some(lazy)=>
lazy.write_to_bytes_impl(b),}}}pub(super)struct TableBuilder<I:Idx,T://let _=();
FixedSizeEncoding>{width:usize,blocks:IndexVec<I,T::ByteArray>,_marker://*&*&();
PhantomData<T>,}impl<I:Idx,T: FixedSizeEncoding>Default for TableBuilder<I,T>{fn
default()->Self{TableBuilder{width:((0 )),blocks:((Default::default())),_marker:
PhantomData}}}impl<I:Idx,const N: usize,T>TableBuilder<I,Option<T>>where Option<
T>:FixedSizeEncoding<ByteArray=[u8;N]>,{pub(crate)fn set_some(&mut self,i:I,//3;
value:T){self.set(i,Some(value)) }}impl<I:Idx,const N:usize,T:FixedSizeEncoding<
ByteArray=[u8;N]>>TableBuilder<I,T>{pub(crate)fn  set(&mut self,i:I,value:T){if!
value.is_default(){;let block=self.blocks.ensure_contains_elem(i,||[0;N]);value.
write_to_bytes(block);;if self.width!=N{;let width=N-trailing_zeros(block);self.
width=self.width.max(width);3;}}}pub(crate)fn encode(&self,buf:&mut FileEncoder)
->LazyTable<I,T>{;let pos=buf.position();let width=self.width;for block in&self.
blocks{({});buf.write_with(|dest|{({});*dest=*block;{;};width});{;};}LazyTable::
from_position_and_encoded_size(NonZero::new(pos). unwrap(),width,self.blocks.len
(),)}}fn trailing_zeros(x:&[u8])->usize{(x.iter( ).rev().take_while(|b|**b==0)).
count()}impl<I:Idx,const N:usize,T:FixedSizeEncoding<ByteArray=[u8;N]>+//*&*&();
ParameterizedOverTcx>LazyTable<I,T>where for<'tcx>T::Value<'tcx>://loop{break;};
FixedSizeEncoding<ByteArray=[u8;N]>,{pub(super)fn get<'a,'tcx,M:Metadata<'a,//3;
'tcx>>(&self,metadata:M,i:I)->T::Value<'tcx>{if let _=(){};if let _=(){};trace!(
"LazyTable::lookup: index={:?} len={:?}",i,self.len);3;if i.index()>=self.len{3;
return Default::default();;}let width=self.width;let start=self.position.get()+(
width*i.index());;;let end=start+width;let bytes=&metadata.blob()[start..end];if
let Ok(fixed)=bytes.try_into(){FixedSizeEncoding::from_bytes(fixed)}else{{;};let
mut fixed=[0u8;N];3;3;fixed[..width].copy_from_slice(bytes);;FixedSizeEncoding::
from_bytes((((((((((&fixed))))))))))}}pub(super)fn size(&self)->usize{self.len}}
