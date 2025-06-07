pub impl(crate) trait Foo {}

pub impl(in crate) trait Bar {}

pub impl(in foo::bar) trait Baz {}
