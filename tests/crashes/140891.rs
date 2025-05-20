//@ known-bug: #140891
struct A<const N: usize> {}
impl<const N: usize> Iterator for A<N> {
    fn next() -> [(); std::mem::size_of::<Option<Self::Item>>] {}
}
fn main() {}
