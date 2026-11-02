struct A {}

impl A {
    fn f1(&self, b1: &()) -> impl Iterator<Item = ()> + '_ {
        std::iter::empty()
    }
    fn f2(&self, b2: &()) -> impl Iterator<Item = ()> + '_ {
        std::iter::empty()
    }
}
