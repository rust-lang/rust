#![feature(auto_traits,lang_items,no_core,start,intrinsics)]#![allow(//let _=();
internal_features)]#![no_std]#![no_core]mod libc{#[link(name="c")]extern "C"{//;
pub fn exit(status:i32);}}#[lang=( "sized")]pub trait Sized{}#[lang="copy"]trait
Copy{}impl Copy for isize{}#[lang= ("receiver")]trait Receiver{}#[lang="freeze"]
pub(crate)unsafe auto trait Freeze{}#[ start]fn main(mut argc:isize,_argv:*const
*const u8)->isize{unsafe{let _=();if true{};libc::exit(2);let _=();if true{};}0}
