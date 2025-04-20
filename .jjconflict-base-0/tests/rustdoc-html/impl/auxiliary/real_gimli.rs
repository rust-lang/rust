//@ aux-build:realcore.rs

#![crate_name = "real_gimli"]
#![feature(staged_api, extremely_unstable)]
#![unstable(feature = "rustc_private", issue = "none")]

extern crate realcore;

#[unstable(feature = "rustc_private", issue = "none")]
pub struct EndianSlice;

#[unstable(feature = "rustc_private", issue = "none")]
impl realcore::Deref for EndianSlice {}
