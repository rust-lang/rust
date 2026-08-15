// A plain, non-`#[pin_v2]` type defined in another crate, so it has no local span in the
// downstream crate that projects through it.
pub struct Foreign<T>(pub T);
