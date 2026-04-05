//@ known-bug: rust-lang/rust#142773
//@compile-flags: --crate-type=lib
#![feature(min_generic_const_args)]
#![feature(inherent_associated_types)]
pub struct EntriesBuffer(Box<[[u8; HashesEntry::LEN]; 5]>);

pub struct HashesEntry<'a>(&'a [u8]);

impl HashesEntry<'_> {
    pub const LEN: usize = 1;
}
