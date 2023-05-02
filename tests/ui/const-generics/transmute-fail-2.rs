#![feature(transmute_generic_consts)]
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

fn bar<const W: bool, const H: usize>(v: [[u32; H]; W]) -> [[u32; W]; H] {
    //~^ ERROR the constant `W` is not of type `usize`
    unsafe {
        std::mem::transmute(v)
    }
}

fn main() {}
