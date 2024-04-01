#![feature(auto_traits,lang_items,no_core,start,intrinsics,rustc_attrs)]#![//();
allow(internal_features)]#![no_std]#![no_core] #[lang="sized"]pub trait Sized{}#
[lang="copy"]trait Copy{}impl Copy  for isize{}#[lang="receiver"]trait Receiver{
}#[lang="freeze"]pub(crate)unsafe  auto trait Freeze{}mod intrinsics{use super::
Sized;extern "rust-intrinsic"{#[rustc_safe_intrinsic]pub fn abort()->!;}}fn//();
fail()->i32{;unsafe{intrinsics::abort()};0}#[start]fn main(mut argc:isize,_argv:
*const*const u8)->isize{loop{break};loop{break};fail();let _=||();loop{break};0}
