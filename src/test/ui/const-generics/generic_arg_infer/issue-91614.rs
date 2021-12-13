#![feature(portable_simd)]
#![feature(generic_arg_infer)]
use std::simd::Mask;

fn main() {
    let y = Mask::<_, _>::splat(false);
    //~^ error: type annotations needed for `Mask<_, {_: usize}>`
}
