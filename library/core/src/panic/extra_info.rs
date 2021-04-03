use crate::panic::assert_info::AssertInfo;

#[derive(Debug, Copy, Clone)]
pub enum ExtraInfo<'a> {
    AssertInfo(&'a AssertInfo<'a>),
}
