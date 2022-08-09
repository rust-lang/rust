/// Are values of a type transmutable into values of another type?
///
/// This trait is implemented on-the-fly by the compiler for types `Src` and `Self` when the bits of
/// any value of type `Self` are safely transmutable into a value of type `Dst`, in a given `Context`,
/// notwithstanding whatever safety checks you have asked the compiler to [`Assume`] are satisfied.
#[unstable(feature = "transmutability", issue = "99571")]
#[lang = "transmute_trait"]
#[rustc_on_unimplemented(
    message = "`{Src}` cannot be safely transmuted into `{Self}` in the defining scope of `{Context}`.",
    label = "`{Src}` cannot be safely transmuted into `{Self}` in the defining scope of `{Context}`."
)]
pub unsafe trait BikeshedIntrinsicFrom<
    Src,
    Context,
    const ASSUME_ALIGNMENT: bool,
    const ASSUME_LIFETIMES: bool,
    const ASSUME_VALIDITY: bool,
    const ASSUME_VISIBILITY: bool,
> where
    Src: ?Sized,
{
}

/// What transmutation safety conditions shall the compiler assume that *you* are checking?
#[unstable(feature = "transmutability", issue = "99571")]
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct Assume {
    /// When `true`, the compiler assumes that *you* are ensuring (either dynamically or statically) that
    /// destination referents do not have stricter alignment requirements than source referents.
    pub alignment: bool,

    /// When `true`, the compiler assume that *you* are ensuring that lifetimes are not extended in a manner
    /// that violates Rust's memory model.
    pub lifetimes: bool,

    /// When `true`, the compiler assumes that *you* are ensuring that the source type is actually a valid
    /// instance of the destination type.
    pub validity: bool,

    /// When `true`, the compiler assumes that *you* have ensured that it is safe for you to violate the
    /// type and field privacy of the destination type (and sometimes of the source type, too).
    pub visibility: bool,
}
