// Test for issue #48463.

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
