//@ known-bug: #101036
#![feature(generic_const_exprs)]

const fn t<const N: usize>() -> u8 {
    N as u8
}

#[repr(u8)]
enum T<const N: u8 = { T::<0>::A as u8 + T::<0>::B as u8 }>
where
    [(); N as usize]:
{
    A = t::<N>() as u8, B
}
