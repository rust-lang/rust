// gate-test-transmute_generic_consts
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

fn transpose<const W: usize, const H: usize>(v: [[u32;H]; W]) -> [[u32; W]; H] {
  unsafe {
    std::mem::transmute(v)
    //~^ ERROR cannot transmute
  }
}

fn ident<const W: usize, const H: usize>(v: [[u32; H]; W]) -> [[u32; H]; W] {
  unsafe {
    std::mem::transmute(v)
  }
}

fn flatten<const W: usize, const H: usize>(v: [[u32; H]; W]) -> [u32; W * H] {
  unsafe {
    std::mem::transmute(v)
    //~^ ERROR cannot transmute
  }
}

fn coagulate<const W: usize, const H: usize>(v: [u32; H*W]) -> [[u32; W];H] {
  unsafe {
    std::mem::transmute(v)
    //~^ ERROR cannot transmute
  }
}

fn flatten_3d<const W: usize, const H: usize, const D: usize>(
  v: [[[u32; D]; H]; W]
) -> [u32; D * W * H] {
  unsafe {
    std::mem::transmute(v)
    //~^ ERROR cannot transmute
  }
}

fn flatten_somewhat<const W: usize, const H: usize, const D: usize>(
  v: [[[u32; D]; H]; W]
) -> [[u32; D * W]; H] {
  unsafe {
    std::mem::transmute(v)
    //~^ ERROR cannot transmute
  }
}

fn known_size<const L: usize>(v: [u16; L]) -> [u8; L * 2] {
  unsafe {
    std::mem::transmute(v)
    //~^ ERROR cannot transmute
  }
}

fn condense_bytes<const L: usize>(v: [u8; L * 2]) -> [u16; L] {
  unsafe {
    std::mem::transmute(v)
    //~^ ERROR cannot transmute
  }
}

fn singleton_each<const L: usize>(v: [u8; L]) -> [[u8;1]; L] {
  unsafe {
    std::mem::transmute(v)
    //~^ ERROR cannot transmute
  }
}

fn transpose_with_const<const W: usize, const H: usize>(
  v: [[u32; 2 * H]; W + W]
) -> [[u32; W + W]; 2 * H] {
  unsafe {
    std::mem::transmute(v)
    //~^ ERROR cannot transmute
  }
}

fn main() {
  let _ = transpose([[0; 8]; 16]);
  let _ = transpose_with_const::<8,4>([[0; 8]; 16]);
  let _ = ident([[0; 8]; 16]);
  let _ = flatten([[0; 13]; 5]);
  let _: [[_; 5]; 13] = coagulate([0; 65]);
  let _ = flatten_3d([[[0; 3]; 13]; 5]);
  let _ = flatten_somewhat([[[0; 3]; 13]; 5]);
  let _ = known_size([16; 13]);
  let _: [u16; 5] = condense_bytes([16u8; 10]);
  let _ = singleton_each([16; 10]);
}
