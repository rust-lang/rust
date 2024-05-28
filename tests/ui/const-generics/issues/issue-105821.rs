//@ failure-status: 101
//@ known-bug: rust-lang/rust#125451
//@ normalize-stderr-test "note: .*\n\n" -> ""
//@ normalize-stderr-test "thread 'rustc' panicked.*\n.*\n" -> ""
//@ normalize-stderr-test "(error: internal compiler error: [^:]+):\d+:\d+: " -> "$1:LL:CC: "
//@ normalize-stderr-test "delayed at .*" -> ""
//@ rustc-env:RUST_BACKTRACE=0

#![allow(incomplete_features)]
#![feature(adt_const_params, generic_const_exprs)]
#![allow(dead_code)]

const fn catone<const M: usize>(_a: &[u8; M]) -> [u8; M + 1]
where
    [(); M + 1]:,
{
    unimplemented!()
}

struct Catter<const A: &'static [u8]>;
impl<const A: &'static [u8]> Catter<A>
where
    [(); A.len() + 1]:,
{
    const ZEROS: &'static [u8; A.len()] = &[0_u8; A.len()];
    const R: &'static [u8] = &catone(Self::ZEROS);
}

fn main() {}
