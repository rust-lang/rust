#![feature(impl_restriction)]

pub
impl(crate)
trait Foo {}

pub impl
( in
crate )
trait
Bar
{}

pub
impl ( in foo
::
bar )
trait Baz {}

pub
impl
(self)
const
trait QuxConst {}

pub
impl(
super
) auto
trait QuxAuto {}

pub
impl
(in crate) unsafe
trait QuxUnsafe {}

pub
impl
(in super
::foo)
const unsafe
trait QuxConstUnsafe {}
