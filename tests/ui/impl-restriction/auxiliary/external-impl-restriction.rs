#![feature(impl_restriction)]
#![expect(incomplete_features)]

pub impl(crate) trait TopLevel {}

pub mod inner {
    pub impl(self) trait Inner {}
}
