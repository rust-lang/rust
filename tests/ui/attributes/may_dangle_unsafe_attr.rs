//@ check-pass
#![crate_type = "lib"]
#![feature(dropck_eyepatch)]

struct PerhapsLoosely<T>(T);

impl<#[unsafe(may_dangle)] T> Drop for PerhapsLoosely<T> {
    fn drop(&mut self) {
        todo!()
    }
}
