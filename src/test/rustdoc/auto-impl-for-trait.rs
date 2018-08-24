// Test for https://github.com/rust-lang/rust/issues/48463 issue.

use std::any::Any;
use std::ops::Deref;

pub struct AnyValue {
    val: Box<Any>,
}

impl Deref for AnyValue {
    type Target = Any;

    fn deref(&self) -> &Any {
        &*self.val
    }
}
