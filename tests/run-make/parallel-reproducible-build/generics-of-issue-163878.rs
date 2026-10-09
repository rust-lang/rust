pub struct A {}

impl A {
    pub fn f1(&self, _b1: &()) -> impl Iterator<Item = ()> + '_ {
        std::iter::empty()
    }
    pub fn f2(&self, _b2: &()) -> impl Iterator<Item = ()> + '_ {
        std::iter::empty()
    }
}
