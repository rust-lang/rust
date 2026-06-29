#![feature(impl_restriction)]

pub impl(crate) trait TopLevel {}

pub mod inner {
    pub impl(self) trait Inner {}
}
