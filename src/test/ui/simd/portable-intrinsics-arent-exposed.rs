// May not matter, since people can use them with a nightly feature.
// However this tests to guarantee they don't leak out via portable_simd,
// and thus don't accidentally get stabilized.
use core::simd::intrinsics; //~ERROR E0433
use std::simd::intrinsics; //~ERROR E0432

fn main() {
    ()
}
