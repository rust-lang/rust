use std::sync::Arc;

pub trait A {
    fn f();
    fn f2(self: &Arc<Self>);
}
