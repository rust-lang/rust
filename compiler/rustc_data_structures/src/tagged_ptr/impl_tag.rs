/// Implements [`Tag`] for a given type.
///
/// You can use `impl_tag` on structs and enums.
/// You need to specify the type and all its possible values,
/// which can only be paths with optional fields.
///
/// [`Tag`]: crate::tagged_ptr::Tag
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use rustc_data_structures::{impl_tag, tagged_ptr::Tag};
///
/// #[derive(Copy, Clone, PartialEq, Debug)]
/// enum SomeTag {
///     A,
///     B,
///     X { v: bool },
///     Y(bool, bool),
/// }
///
/// impl_tag! {
///     // The type for which the `Tag` will be implemented
///     impl Tag for SomeTag;
///     // You need to specify the `{value_of_the_type} <=> {tag}` relationship
///     SomeTag::A <=> 0,
///     SomeTag::B <=> 1,
///     // For variants with fields, you need to specify the fields:
///     SomeTag::X { v: true  } <=> 2,
///     SomeTag::X { v: false } <=> 3,
///     // For tuple variants use named syntax:
///     SomeTag::Y { 0: true,  1: true  } <=> 4,
///     SomeTag::Y { 0: false, 1: true  } <=> 5,
///     SomeTag::Y { 0: true,  1: false } <=> 6,
///     SomeTag::Y { 0: false, 1: false } <=> 7,
/// }
///
/// assert_eq!(SomeTag::A.into_usize(), 0);
/// assert_eq!(SomeTag::X { v: false }.into_usize(), 3);
/// assert_eq!(SomeTag::Y(false, true).into_usize(), 5);
///
/// assert_eq!(unsafe { SomeTag::from_usize(1) }, SomeTag::B);
/// assert_eq!(unsafe { SomeTag::from_usize(2) }, SomeTag::X { v: true });
/// assert_eq!(unsafe { SomeTag::from_usize(7) }, SomeTag::Y(false, false));
/// ```
///
/// Structs are supported:
///
/// ```
/// # use rustc_data_structures::impl_tag;
/// #[derive(Copy, Clone)]
/// struct Flags { a: bool, b: bool }
///
/// impl_tag! {
///     impl Tag for Flags;
///     Flags { a: true,  b: true  } <=> 3,
///     Flags { a: false, b: true  } <=> 2,
///     Flags { a: true,  b: false } <=> 1,
///     Flags { a: false, b: false } <=> 0,
/// }
/// ```
///
/// Not specifying all values results in a compile error:
///
/// ```compile_fail,E0004
/// # use rustc_data_structures::impl_tag;
/// #[derive(Copy, Clone)]
/// enum E {
///     A,
///     B,
/// }
///
/// impl_tag! {
///     impl Tag for E;
///     E::A <=> 0,
/// }
/// ```
#[macro_export]
macro_rules! impl_tag {
    (
        impl Tag for $Self:ty;
        $(
            $($path:ident)::* $( { $( $fields:tt )* })? <=> $tag:literal,
        )*
    ) => {
        // Safety:
        // `into_usize` only returns one of `$tag`s,
        // `bits_for_tags` is called on all `$tag`s,
        // thus `BITS` constant is correct.
        unsafe impl $crate::tagged_ptr::Tag for $Self {
            const BITS: u32 = $crate::tagged_ptr::bits_for_tags(&[
                $( $tag, )*
            ]);

            fn into_usize(self) -> usize {
                // This forbids use of repeating patterns (`Enum::V`&`Enum::V`, etc)
                // (or at least it should, see <https://github.com/rust-lang/rust/issues/110613>)
                #[forbid(unreachable_patterns)]
                match self {
                    // `match` is doing heavy lifting here, by requiring exhaustiveness
                    $(
                        $($path)::* $( { $( $fields )* } )? => $tag,
                    )*
                }
            }

            unsafe fn from_usize(tag: usize) -> Self {
                // Similarly to the above, this forbids repeating tags
                // (or at least it should, see <https://github.com/rust-lang/rust/issues/110613>)
                #[forbid(unreachable_patterns)]
                match tag {
                    $(
                        $tag => $($path)::* $( { $( $fields )* } )?,
                    )*

                    // Safety:
                    // `into_usize` only returns one of `$tag`s,
                    // all `$tag`s are filtered up above,
                    // thus if this is reached, the safety contract of this
                    // function was already breached.
                    _ => unsafe {
                        debug_assert!(
                            false,
                            "invalid tag: {tag}\
                             (this is a bug in the caller of `from_usize`)"
                        );
                        std::hint::unreachable_unchecked()
                    },
                }
            }

        }
    };
}

#[cfg(test)]
mod tests;
