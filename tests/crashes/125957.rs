//@ known-bug: rust-lang/rust#125957
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
#![feature(associated_const_equality)]

pub struct Equal<const T: usize, const R: usize>();

pub enum ParseMode {
    Raw,
}
pub trait Parse {
    const PARSE_MODE: ParseMode;
}
pub trait RenderRaw: Parse<PARSE_MODE = { ParseMode::Raw }> {}

trait GenericVec<T> {
    fn unwrap() -> dyn RenderRaw;
}

fn main() {}
