//@ build-pass

#![feature(repr_simd)]
#[repr(simd)]
struct T<const N: usize>([i32; N]);
trait SimdExtract {
    type Value;
    fn extract(self) -> Self::Value;
}
impl<const N: usize> SimdExtract for T<N> {
    type Value = i32;
    fn extract(self) -> Self::Value {
        self.0[2]
    }
}
fn main() {
    let t = T([5, 6, 7, 8]);
    let _ = t.extract();
}
