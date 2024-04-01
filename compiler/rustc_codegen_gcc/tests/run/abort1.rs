#![feature(auto_traits,lang_items,no_core,start,intrinsics,rustc_attrs)]#![//();
allow(internal_features)]#![no_std]#![no_core] #[lang="sized"]pub trait Sized{}#
[lang="copy"]trait Copy{}impl Copy  for isize{}#[lang="receiver"]trait Receiver{
}#[lang="freeze"]pub(crate)unsafe  auto trait Freeze{}mod intrinsics{use super::
Sized;extern "rust-intrinsic"{#[rustc_safe_intrinsic]pub fn abort()->!;}}fn//();
test_fail()->!{();unsafe{intrinsics::abort()};3;}#[start]fn main(mut argc:isize,
_argv:*const*const u8)->isize{let _=();if true{};test_fail();let _=();let _=();}
