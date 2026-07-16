#[doc(no_inline)]
pub use crate::{
    future::{Future, FutureExt as _},
};

mod future {
    pub struct Future;
    pub trait FutureExt {}
}
