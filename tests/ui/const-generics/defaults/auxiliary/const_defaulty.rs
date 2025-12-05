pub struct Defaulted<const N: usize=3>;
impl Defaulted {
    pub fn new() -> Self {
        Defaulted
    }
}
impl<const N: usize> Defaulted<N> {
    pub fn value(&self) -> usize {
        N
    }
}
