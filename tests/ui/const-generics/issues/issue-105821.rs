#![allow(incomplete_features)]
#![feature(adt_const_params, const_ptr_read, generic_const_exprs)]
#![allow(dead_code)]

const fn catone<const M: usize>(_a: &[u8; M]) -> [u8; M + 1]
where
    [(); M + 1]:,
{
  [0; M+1]
}

struct Catter<const A: &'static [u8]>;
impl<const A: &'static [u8]> Catter<A>
where
    [(); A.len() + 1]:,
{
    const ZEROS: &'static [u8; A.len()] = &[0_u8; A.len()];
    //~^ ERROR overly complex
    const R: &'static [u8] = &catone(Self::ZEROS);
    //~^ ERROR overly complex
}

fn main() {
  let _ = Catter::<{&[]}>::R;
  let _ = Catter::<{&[]}>::ZEROS;
}
