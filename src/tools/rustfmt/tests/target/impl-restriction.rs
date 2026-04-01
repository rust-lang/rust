#![feature(impl_restriction)]

pub impl(crate) trait Foo {}

pub impl(in crate) trait Bar {}

pub impl(in foo::bar) trait Baz {}

pub const impl(self) trait QuxConst {}

pub auto impl(super) trait QuxAuto {}

pub unsafe impl(in crate) trait QuxUnsafe {}

pub const unsafe impl(in super::foo) trait QuxConstUnsafe {}
