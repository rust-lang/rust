use crate::panic::assert_info::AssertInfo;

#[derive(Debug, Copy, Clone)]
#[non_exhaustive]
pub enum ExtraInfo<'a> {
    AssertInfo(&'a AssertInfo<'a>),
}
