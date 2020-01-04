/// "Signaling" trait used in impl trait to tag lifetimes that you may
/// need to capture but don't really need for other reasons.
/// Basically a workaround; see [this comment] for details.
///
/// [this comment]: https://github.com/rust-lang/rust/issues/34511#issuecomment-373423999
// FIXME(eddyb) false positive, the lifetime parameter is "phantom" but needed.
#[allow(unused_lifetimes)]
pub trait Captures<'a> {}

impl<'a, T: ?Sized> Captures<'a> for T {}

// FIXME(eddyb) false positive, the lifetime parameter is "phantom" but needed.
#[allow(unused_lifetimes)]
pub trait Captures2<'a, 'b> {}

impl<'a, 'b, T: ?Sized> Captures2<'a, 'b> for T {}
