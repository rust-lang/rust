fn a() {}
pub fn b() {}
pub macro m($:ident) {}
pub(crate) fn c() {}
pub(super) fn d() {}
pub(in foo::bar::baz) fn e() {}
