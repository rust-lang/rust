//@ run-pass
#![feature(generic_const_exprs)]
#![feature(transmute_generic_consts)]
#![allow(incomplete_features)]

fn ident<const W: usize, const H: usize>(v: [[u32; H]; W]) -> [[u32; H]; W] {
  unsafe {
    std::mem::transmute(v)
  }
}

fn main() {
  let _ = ident([[0; 8]; 16]);
}
