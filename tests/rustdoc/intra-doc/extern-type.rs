#![feature(extern_types)]

extern {
    pub type ExternType;
}

pub trait T {
    fn test(&self) {}
}

pub trait G<N> {
    fn g(&self, n: N) {}
}

impl ExternType {
    pub fn f(&self) {}
}

impl T for ExternType {
    fn test(&self) {}
}

impl G<usize> for ExternType {
    fn g(&self, n: usize) {}
}

//@ has 'extern_type/externtype.ExternType.html'
//@ hasraw 'extern_type/fn.links_to_extern_type.html' \
// 'href="externtype.ExternType.html#method.f"'
//@ hasraw 'extern_type/fn.links_to_extern_type.html' \
// 'href="externtype.ExternType.html#method.test"'
//@ hasraw 'extern_type/fn.links_to_extern_type.html' \
// 'href="externtype.ExternType.html#method.g"'
/// See also [ExternType::f]
/// See also [ExternType::test]
/// See also [ExternType::g]
pub fn links_to_extern_type() {}
