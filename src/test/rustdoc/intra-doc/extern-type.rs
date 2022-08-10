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

// @has 'extern_type/foreigntype.ExternType.html'
// @hastext 'extern_type/fn.links_to_extern_type.html' \
// 'href="foreigntype.ExternType.html#method.f"'
// @hastext 'extern_type/fn.links_to_extern_type.html' \
// 'href="foreigntype.ExternType.html#method.test"'
// @hastext 'extern_type/fn.links_to_extern_type.html' \
// 'href="foreigntype.ExternType.html#method.g"'
/// See also [ExternType::f]
/// See also [ExternType::test]
/// See also [ExternType::g]
pub fn links_to_extern_type() {}
