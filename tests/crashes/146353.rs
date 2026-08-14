//@ known-bug: rust-lang/rust#146353
#![feature(trait_alias)]

use std::mem::{MaybeUninit};
const BIG_CHAIN = MaybeUninit::uninit();

pub trait NeverSend = !Send;
