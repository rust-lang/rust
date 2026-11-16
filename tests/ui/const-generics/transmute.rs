//@ run-pass
#![feature(generic_const_exprs)]
#![feature(transmute_generic_consts)]
#![allow(incomplete_features)]

fn ident<const W: usize, const H: usize>(v: [[u32; H]; W]) -> [[u32; H]; W] {
    unsafe { std::mem::transmute(v) }
}

fn singleton_each<const L: usize>(v: [u8; L]) -> [[u8; 1]; L] {
    unsafe { std::mem::transmute(v) }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct T(u8);

fn element<const N: usize>(v: [T; N]) -> [u8; N] {
    unsafe { std::mem::transmute::<_, _>(v) }
}

fn element2<const N: usize, const M: usize>(v: [[T; N]; M]) -> [[u8; N]; M] {
    unsafe { std::mem::transmute::<_, _>(v) }
}

fn element3_mixed<const N: usize, const M: usize>(v: [[[T; N]; 42]; M]) -> [[[u8; N]; 42]; M] {
    unsafe { std::mem::transmute::<_, _>(v) }
}

fn element4_swap_lit<const N: usize, const M: usize>(
    v: [[[[T; N]; 3]; 7]; M],
) -> [[[[u8; 7]; N]; M]; 3] {
    unsafe { std::mem::transmute::<_, _>(v) }
}

fn main() {
    let _ = ident([[0; 8]; 16]);
    let _ = singleton_each([0; 8]);
    let _ = element([T(0); 8]);
    let _ = element2([[T(0); 8]; 16]);
    let _ = element3_mixed([[[T(0); 8]; 42]; 16]);
    let _ = element4_swap_lit([[[[T(0); 8]; 3]; 7]; 16]);
}
