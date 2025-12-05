//@ check-pass
// If this test starts failing because it ICEs due to not being able to convert a `ReErased` to
// something then feel free to just convert this to a known-bug. I'm pretty sure this is still
// a failing test, we just started masking the bug.

#![allow(incomplete_features)]
#![feature(adt_const_params, unsized_const_params, generic_const_exprs)]
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
