#![crate_type = "lib"]
#![feature(dropck_eyepatch)]

struct PerhapsLoosely<T>(T);

unsafe impl<#[unsafe(may_dangle)] T> Drop for PerhapsLoosely<T> {
    //~^ERROR implementing the `Drop` trait is not unsafe
    fn drop(&mut self) {
        todo!()
    }
}
