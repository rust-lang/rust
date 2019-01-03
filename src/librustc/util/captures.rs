/// "Signaling" trait used in impl trait to tag lifetimes that you may
/// need to capture but don't really need for other reasons.
/// Basically a workaround; see [this comment] for details.
///
/// [this comment]: https://github.com/rust-lang/rust/issues/34511#issuecomment-373423999
pub trait Captures<'a> {}

impl<'a, T: ?Sized> Captures<'a> for T {}
