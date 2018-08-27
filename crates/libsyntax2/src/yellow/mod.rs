mod builder;
mod green;
mod red;
mod syntax;

use std::{
    sync::Arc,
    ptr,
};
pub use self::syntax::{SyntaxNode, SyntaxNodeRef, SyntaxError};
pub(crate) use self::{
    builder::GreenBuilder,
    green::GreenNode,
    red::RedNode,
};

#[derive(Debug)]
pub struct SyntaxRoot {
    red: RedNode,
    pub(crate) errors: Vec<SyntaxError>,
}

pub trait TreeRoot: Clone + Send + Sync {
    fn borrowed(&self) -> RefRoot;
    fn owned(&self) -> OwnedRoot;

    #[doc(hidden)]
    fn syntax_root(&self) -> &SyntaxRoot;
}
#[derive(Clone, Debug)]
pub struct OwnedRoot(Arc<SyntaxRoot>);
#[derive(Clone, Copy, Debug)]
pub struct RefRoot<'a>(&'a OwnedRoot); // TODO: shared_from_this instead of double indirection

impl TreeRoot for OwnedRoot {
    fn borrowed(&self) -> RefRoot {
        RefRoot(&self)
    }
    fn owned(&self) -> OwnedRoot {
        self.clone()
    }

    fn syntax_root(&self) -> &SyntaxRoot {
        &*self.0
    }
}

impl<'a> TreeRoot for RefRoot<'a> {
    fn borrowed(&self) -> RefRoot {
        *self
    }
    fn owned(&self) -> OwnedRoot {
        self.0.clone()
    }
    fn syntax_root(&self) -> &SyntaxRoot {
        self.0.syntax_root()
    }
}

impl SyntaxRoot {
    pub(crate) fn new(green: GreenNode, errors: Vec<SyntaxError>) -> SyntaxRoot {
        SyntaxRoot {
            red: RedNode::new_root(green),
            errors,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
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
