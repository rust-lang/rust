#![feature(transmute_generic_consts)]
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

fn foo<const W: usize, const H: usize>(v: [[u32;H+1]; W]) -> [[u32; W+1]; H] {
  unsafe {
    std::mem::transmute(v)
    //~^ ERROR cannot transmute
  }
}

fn bar<const W: bool, const H: usize>(v: [[u32; H]; W]) -> [[u32; W]; H] {
  //~^ ERROR mismatched types
  //~| ERROR mismatched types
  unsafe {
    std::mem::transmute(v)
    //~^ ERROR cannot transmute between types
  }
}

fn baz<const W: usize, const H: usize>(v: [[u32; H]; W]) -> [u32; W * H * H] {
  unsafe {
    std::mem::transmute(v)
    //~^ ERROR cannot transmute
  }
}

fn overflow(v: [[[u32; 8888888]; 9999999]; 777777777]) -> [[[u32; 9999999]; 777777777]; 8888888] {
  unsafe {
    std::mem::transmute(v)
    //~^ ERROR cannot transmute
  }
}

fn main() {}
