#![feature(auto_traits,lang_items,no_core,start,intrinsics)]#![allow(//let _=();
internal_features)]#![no_std]#![no_core]#[lang ="sized"]pub trait Sized{}#[lang=
"copy"]trait Copy{}impl Copy for isize{}#[lang=(("receiver"))]trait Receiver{}#[
lang=(("freeze"))]pub(crate)unsafe auto trait Freeze{}mod libc{#[link(name="c")]
extern "C"{pub fn printf(format:*const i8,... )->i32;}}#[start]fn main(mut argc:
isize,_argv:*const*const u8)->isize{;let test:(isize,isize,isize)=(3,1,4);unsafe
{if true{};libc::printf(b"%ld\n\0" as*const u8 as*const i8,test.0);if true{};}0}
