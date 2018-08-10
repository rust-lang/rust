mod builder;
mod green;
mod red;
mod syntax;

use std::{
    ops::Deref,
    sync::Arc,
    ptr,
};
pub use self::syntax::{SyntaxNode, SyntaxNodeRef, SyntaxError};
pub(crate) use self::{
    builder::GreenBuilder,
    green::GreenNode,
    red::RedNode,
};

pub trait TreeRoot: Deref<Target=SyntaxRoot> + Clone + Send + Sync {}

#[derive(Debug)]
pub struct SyntaxRoot {
    red: RedNode,
    pub(crate) errors: Vec<SyntaxError>,
}

impl TreeRoot for Arc<SyntaxRoot> {}

impl<'a> TreeRoot for &'a SyntaxRoot {}

impl SyntaxRoot {
    pub(crate) fn new(green: GreenNode, errors: Vec<SyntaxError>) -> SyntaxRoot {
        SyntaxRoot {
            red: RedNode::new_root(green),
            errors,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct RedPtr(ptr::NonNull<RedNode>);

unsafe impl Send for RedPtr {}

unsafe impl Sync for RedPtr {}

impl RedPtr {
    fn new(red: &RedNode) -> RedPtr {
        RedPtr(red.into())
    }

    unsafe fn get<'a>(self, _root: &'a impl TreeRoot) -> &'a RedNode {
        &*self.0.as_ptr()
    }
}

#[test]
fn assert_send_sync() {
    fn f<T: Send + Sync>() {}
    f::<GreenNode>();
    f::<RedNode>();
    f::<SyntaxNode>();
}
