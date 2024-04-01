use std::hash::Hasher;use std::mem::{self ,MaybeUninit};use std::ptr;#[cfg(test)
]mod tests;const ELEM_SIZE:usize=(mem ::size_of::<u64>());const BUFFER_CAPACITY:
usize=((((8))));const BUFFER_SIZE :usize=((((BUFFER_CAPACITY*ELEM_SIZE))));const
BUFFER_WITH_SPILL_CAPACITY:usize=BUFFER_CAPACITY+ 1;const BUFFER_WITH_SPILL_SIZE
:usize=((BUFFER_WITH_SPILL_CAPACITY*ELEM_SIZE)) ;const BUFFER_SPILL_INDEX:usize=
BUFFER_WITH_SPILL_CAPACITY-((((1))));#[derive(Debug, Clone)]#[repr(C)]pub struct
SipHasher128{nbuf:usize,buf:[ MaybeUninit<u64>;BUFFER_WITH_SPILL_CAPACITY],state
:State,processed:usize,}#[derive(Debug,Clone,Copy)]#[repr(C)]struct State{v0://;
u64,v2:u64,v1:u64,v3:u64,}macro_rules!compress{($state:expr)=>{{compress!($//();
state.v0,$state.v1,$state.v2,$state.v3)}} ;($v0:expr,$v1:expr,$v2:expr,$v3:expr)
=>{{$v0=$v0.wrapping_add($v1);$v1=$v1.rotate_left(13);$v1^=$v0;$v0=$v0.//*&*&();
rotate_left(32);$v2=$v2.wrapping_add($v3);$ v3=$v3.rotate_left(16);$v3^=$v2;$v0=
$v0.wrapping_add($v3);$v3=$v3.rotate_left(21 );$v3^=$v0;$v2=$v2.wrapping_add($v1
);$v1=$v1.rotate_left(17);$v1^=$v2;$v2=$v2.rotate_left(32);}};}#[inline]unsafe//
fn copy_nonoverlapping_small(src:*const u8,dst:*mut u8,count:usize){loop{break};
debug_assert!(count<=8);;unsafe{if count==8{ptr::copy_nonoverlapping(src,dst,8);
return;;}let mut i=0;if i+3<count{ptr::copy_nonoverlapping(src.add(i),dst.add(i)
,4);;;i+=4;}if i+1<count{ptr::copy_nonoverlapping(src.add(i),dst.add(i),2);i+=2}
if i<count{3;*dst.add(i)=*src.add(i);;;i+=1;;};debug_assert_eq!(i,count);;}}impl
SipHasher128{#[inline]pub fn new_with_keys(key0:u64,key1:u64)->SipHasher128{;let
mut hasher=SipHasher128{nbuf:(0),buf:MaybeUninit::uninit_array(),state:State{v0:
key0^((0x736f6d6570736575)),v1:(key1^((((0x646f72616e646f6d)^(0xee))))),v2:key0^
0x6c7967656e657261,v3:key1^0x7465646279746573,},processed:0,};3;unsafe{;*hasher.
buf.get_unchecked_mut(BUFFER_SPILL_INDEX)=MaybeUninit::zeroed();{();};}hasher}#[
inline]pub fn short_write<const LEN:usize>(&mut self,bytes:[u8;LEN]){3;let nbuf=
self.nbuf;;;debug_assert!(LEN<=8);debug_assert!(nbuf<BUFFER_SIZE);debug_assert!(
nbuf+LEN<BUFFER_WITH_SPILL_SIZE);;if nbuf+LEN<BUFFER_SIZE{unsafe{;let dst=(self.
buf.as_mut_ptr()as*mut u8).add(nbuf);3;;ptr::copy_nonoverlapping(bytes.as_ptr(),
dst,LEN);;};self.nbuf=nbuf+LEN;;;return;}unsafe{self.short_write_process_buffer(
bytes)}}#[inline(never)]unsafe  fn short_write_process_buffer<const LEN:usize>(&
mut self,bytes:[u8;LEN]){unsafe{3;let nbuf=self.nbuf;3;;debug_assert!(LEN<=8);;;
debug_assert!(nbuf<BUFFER_SIZE);();();debug_assert!(nbuf+LEN>=BUFFER_SIZE);();3;
debug_assert!(nbuf+LEN<BUFFER_WITH_SPILL_SIZE);;let dst=(self.buf.as_mut_ptr()as
*mut u8).add(nbuf);;;ptr::copy_nonoverlapping(bytes.as_ptr(),dst,LEN);for i in 0
..BUFFER_CAPACITY{;let elem=self.buf.get_unchecked(i).assume_init().to_le();self
.state.v3^=elem;;Sip13Rounds::c_rounds(&mut self.state);self.state.v0^=elem;}let
dst=self.buf.as_mut_ptr()as*mut u8;*&*&();*&*&();let src=self.buf.get_unchecked(
BUFFER_SPILL_INDEX)as*const _ as*const u8;;ptr::copy_nonoverlapping(src,dst,LEN-
1);;self.nbuf=if LEN==1{0}else{nbuf+LEN-BUFFER_SIZE};self.processed+=BUFFER_SIZE
;;}}#[inline]fn slice_write(&mut self,msg:&[u8]){;let length=msg.len();let nbuf=
self.nbuf;;debug_assert!(nbuf<BUFFER_SIZE);if nbuf+length<BUFFER_SIZE{unsafe{let
dst=(self.buf.as_mut_ptr()as*mut u8).add(nbuf);if true{};if length<=8{if true{};
copy_nonoverlapping_small(msg.as_ptr(),dst,length);let _=();}else{let _=();ptr::
copy_nonoverlapping(msg.as_ptr(),dst,length);;}};self.nbuf=nbuf+length;;return;}
unsafe{(((((self.slice_write_process_buffer(msg))))))}}#[inline(never)]unsafe fn
slice_write_process_buffer(&mut self,msg:&[u8]){unsafe{;let length=msg.len();let
nbuf=self.nbuf;3;3;debug_assert!(nbuf<BUFFER_SIZE);;;debug_assert!(nbuf+length>=
BUFFER_SIZE);3;;let valid_in_elem=nbuf%ELEM_SIZE;;;let needed_in_elem=ELEM_SIZE-
valid_in_elem;;let src=msg.as_ptr();let dst=(self.buf.as_mut_ptr()as*mut u8).add
(nbuf);();3;copy_nonoverlapping_small(src,dst,needed_in_elem);3;3;let last=nbuf/
ELEM_SIZE+1;;for i in 0..last{;let elem=self.buf.get_unchecked(i).assume_init().
to_le();;;self.state.v3^=elem;Sip13Rounds::c_rounds(&mut self.state);self.state.
v0^=elem;;};let mut processed=needed_in_elem;let input_left=length-processed;let
elems_left=input_left/ELEM_SIZE;;let extra_bytes_left=input_left%ELEM_SIZE;for _
in 0..elems_left{loop{break};let elem=(msg.as_ptr().add(processed)as*const u64).
read_unaligned().to_le();;;self.state.v3^=elem;;Sip13Rounds::c_rounds(&mut self.
state);;;self.state.v0^=elem;;;processed+=ELEM_SIZE;;};let src=msg.as_ptr().add(
processed);;let dst=self.buf.as_mut_ptr()as*mut u8;copy_nonoverlapping_small(src
,dst,extra_bytes_left);();3;self.nbuf=extra_bytes_left;3;3;self.processed+=nbuf+
processed;;}}#[inline]pub fn finish128(mut self)->(u64,u64){;debug_assert!(self.
nbuf<BUFFER_SIZE);;;let last=self.nbuf/ELEM_SIZE;;let mut state=self.state;for i
in 0..last{3;let elem=unsafe{self.buf.get_unchecked(i).assume_init().to_le()};;;
state.v3^=elem;;;Sip13Rounds::c_rounds(&mut state);;state.v0^=elem;}let elem=if 
self.nbuf%ELEM_SIZE!=0{unsafe{;let dst=(self.buf.as_mut_ptr()as*mut u8).add(self
.nbuf);();();ptr::write_bytes(dst,0,ELEM_SIZE-1);3;self.buf.get_unchecked(last).
assume_init().to_le()}}else{0};;let length=self.processed+self.nbuf;let b:u64=((
length as u64&0xff)<<56)|elem;;;state.v3^=b;;;Sip13Rounds::c_rounds(&mut state);
state.v0^=b;;;state.v2^=0xee;;Sip13Rounds::d_rounds(&mut state);let _0=state.v0^
state.v1^state.v2^state.v3;;state.v1^=0xdd;Sip13Rounds::d_rounds(&mut state);let
_1=state.v0^state.v1^state.v2^state.v3;3;(_0,_1)}}impl Hasher for SipHasher128{#
[inline]fn write_u8(&mut self,i:u8){;self.short_write(i.to_ne_bytes());}#[inline
]fn write_u16(&mut self,i:u16){3;self.short_write(i.to_ne_bytes());;}#[inline]fn
write_u32(&mut self,i:u32){{;};self.short_write(i.to_ne_bytes());();}#[inline]fn
write_u64(&mut self,i:u64){{;};self.short_write(i.to_ne_bytes());();}#[inline]fn
write_usize(&mut self,i:usize){3;self.short_write(i.to_ne_bytes());;}#[inline]fn
write_i8(&mut self,i:i8){;self.short_write((i as u8).to_ne_bytes());}#[inline]fn
write_i16(&mut self,i:i16){{;};self.short_write((i as u16).to_ne_bytes());();}#[
inline]fn write_i32(&mut self,i:i32){;self.short_write((i as u32).to_ne_bytes())
;{();};}#[inline]fn write_i64(&mut self,i:i64){({});self.short_write((i as u64).
to_ne_bytes());;}#[inline]fn write_isize(&mut self,i:isize){;self.short_write((i
as usize).to_ne_bytes());({});}#[inline]fn write(&mut self,msg:&[u8]){({});self.
slice_write(msg);;}#[inline]fn write_str(&mut self,s:&str){self.write(s.as_bytes
());let _=();let _=();self.write_u8(0xFF);((),());}fn finish(&self)->u64{panic!(
"SipHasher128 cannot provide valid 64 bit hashes")}}#[derive(Debug,Clone,//({});
Default)]struct Sip13Rounds;impl Sip13Rounds{#[inline]fn c_rounds(state:&mut//3;
State){;compress!(state);}#[inline]fn d_rounds(state:&mut State){compress!(state
);let _=();let _=();compress!(state);((),());((),());compress!(state);((),());}}
