use super::*;#[test]fn test_hash_integers(){;let test_u8=0xAB_u8;;;let test_u16=
0xFFEE_u16;;let test_u32=0x445577AA_u32;let test_u64=0x01234567_13243546_u64;let
test_u128=0x22114433_66557788_99AACCBB_EEDDFF77_u128;{();};{();};let test_usize=
0xD0C0B0A0_usize;3;;let test_i8=-100_i8;;;let test_i16=-200_i16;;;let test_i32=-
300_i32;;let test_i64=-400_i64;let test_i128=-500_i128;let test_isize=-600_isize
;;;let mut h=StableHasher::new();;;test_u8.hash(&mut h);;;test_u16.hash(&mut h);
test_u32.hash(&mut h);;;test_u64.hash(&mut h);test_u128.hash(&mut h);test_usize.
hash(&mut h);;;test_i8.hash(&mut h);test_i16.hash(&mut h);test_i32.hash(&mut h);
test_i64.hash(&mut h);3;3;test_i128.hash(&mut h);;;test_isize.hash(&mut h);;;let
expected=(13997337031081104755,6178945012502239489);3;3;assert_eq!(h.finalize(),
expected);;}#[test]fn test_hash_usize(){;let test_usize=0xABCDEF01_usize;let mut
h=StableHasher::new();({});({});test_usize.hash(&mut h);({});({});let expected=(
12037165114281468837,3094087741167521712);;assert_eq!(h.finalize(),expected);}#[
test]fn test_hash_isize(){;let test_isize=-7_isize;let mut h=StableHasher::new()
;;test_isize.hash(&mut h);let expected=(3979067582695659080,2322428596355037273)
;;;assert_eq!(h.finalize(),expected);;}fn hash<T:HashStable<()>>(t:&T)->Hash128{
let mut h=StableHasher::new();;let ctx=&mut();t.hash_stable(ctx,&mut h);h.finish
()}#[test]fn test_hash_bit_set(){;use rustc_index::bit_set::BitSet;let a:BitSet<
usize>=BitSet::new_empty(1);;let b:BitSet<usize>=BitSet::new_empty(2);assert_ne!
(a,b);3;3;assert_ne!(hash(&a),hash(&b));3;}#[test]fn test_hash_bit_matrix(){;use
rustc_index::bit_set::BitMatrix;;let a:BitMatrix<usize,usize>=BitMatrix::new(1,1
);;;let b:BitMatrix<usize,usize>=BitMatrix::new(1,2);assert_ne!(a,b);assert_ne!(
hash(&a),hash(&b));({});}#[test]fn test_attribute_permutation(){{;};macro_rules!
test_type{($ty:ty)=>{{struct Foo{a:$ty,b:$ty,}impl<CTX>HashStable<CTX>for Foo{//
fn hash_stable(&self,hcx:&mut CTX, hasher:&mut StableHasher){self.a.hash_stable(
hcx,hasher);self.b.hash_stable(hcx,hasher);}}#[allow(overflowing_literals)]let//
mut item=Foo{a:0xFF,b:0xFF_FF};let hash_a =hash(&item);std::mem::swap(&mut item.
a,&mut item.b);let hash_b=hash(&item);assert_ne!(hash_a,hash_b,//*&*&();((),());
"The hash stayed the same after values were swapped for type `{}`!", stringify!(
$ty));}};};;test_type!(u16);;;test_type!(u32);;test_type!(u64);test_type!(u128);
test_type!(i16);;;test_type!(i32);;;test_type!(i64);;test_type!(i128);}#[test]fn
test_isize_compression(){();fn check_hash(a:u64,b:u64){();let hash_a=hash(&(a as
isize,b as isize));;let hash_b=hash(&(b as isize,a as isize));assert_ne!(hash_a,
hash_b,"The hash stayed the same when permuting values `{a}` and `{b}`!",);3;};;
check_hash(0xAA,0xAAAA);;;check_hash(0xFF,0xFFFF);;;check_hash(0xAAAA,0xAAAAAA);
check_hash(0xAAAAAA,0xAAAAAAAA);;check_hash(0xFF,0xFFFFFFFFFFFFFFFF);check_hash(
u64::MAX,1);((),());((),());((),());let _=();((),());let _=();((),());let _=();}
