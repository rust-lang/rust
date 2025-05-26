#![unstable(feature = "phantom_variance_markers", issue = "135806")]

use super::PhantomData;
use crate::any::type_name;
use crate::cmp::Ordering;
use crate::fmt;
use crate::hash::{Hash, Hasher};

macro_rules! first_token {
    ($first:tt $($rest:tt)*) => {
        $first
    };
}

macro_rules! phantom_type {
    ($(
        $(#[$attr:meta])*
        pub struct $name:ident <$t:ident> ($($inner:tt)*);
    )*) => {$(
        $(#[$attr])*
        pub struct $name<$t>($($inner)*) where T: ?Sized;

        impl<T> $name<T>
            where T: ?Sized
        {
            /// Constructs a new instance of the variance marker.
            pub const fn new() -> Self {
                Self(PhantomData)
            }
        }

        impl<T> self::sealed::Sealed for $name<T> where T: ?Sized {
            const VALUE: Self = Self::new();
        }
        impl<T> Variance for $name<T> where T: ?Sized {}

        impl<T> Default for $name<T>
            where T: ?Sized
        {
            fn default() -> Self {
                Self(PhantomData)
            }
        }

        impl<T> fmt::Debug for $name<T>
            where T: ?Sized
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}<{}>", stringify!($name), type_name::<T>())
            }
        }

        impl<T> Clone for $name<T>
            where T: ?Sized
        {
            fn clone(&self) -> Self {
                *self
            }
        }

        impl<T> Copy for $name<T> where T: ?Sized {}

        impl<T> PartialEq for $name<T>
            where T: ?Sized
        {
            fn eq(&self, _: &Self) -> bool {
                true
            }
        }

        impl<T> Eq for $name<T> where T: ?Sized {}

        impl<T> PartialOrd for $name<T>
            where T: ?Sized
        {
            fn partial_cmp(&self, _: &Self) -> Option<Ordering> {
                Some(Ordering::Equal)
            }
        }

        impl<T> Ord for $name<T>
            where T: ?Sized
        {
            fn cmp(&self, _: &Self) -> Ordering {
                Ordering::Equal
            }
        }

        impl<T> Hash for $name<T>
            where T: ?Sized
        {
            fn hash<H: Hasher>(&self, _: &mut H) {}
        }
    )*};
}

macro_rules! phantom_lifetime {
    ($(
        $(#[$attr:meta])*
        pub struct $name:ident <$lt:lifetime> ($($inner:tt)*);
    )*) => {$(
        $(#[$attr])*
        #[derive(Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name<$lt>($($inner)*);

        impl $name<'_> {
            /// Constructs a new instance of the variance marker.
            pub const fn new() -> Self {
                Self(first_token!($($inner)*)(PhantomData))
            }
        }

        impl self::sealed::Sealed for $name<'_> {
            const VALUE: Self = Self::new();
        }
        impl Variance for $name<'_> {}

        impl fmt::Debug for $name<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", stringify!($name))
            }
        }
    )*};
}

phantom_lifetime! {
    /// Zero-sized type used to mark a lifetime as covariant.
    ///
    /// Covariant lifetimes must live at least as long as declared. See [the reference][1] for more
    /// information.
    ///
    /// [1]: https://doc.rust-lang.org/stable/reference/subtyping.html#variance
    ///
    /// Note: If `'a` is otherwise contravariant or invariant, the resulting type is invariant.
    ///
    /// ## Layout
    ///
    /// For all `'a`, the following are guaranteed:
    /// * `size_of::<PhantomCovariantLifetime<'a>>() == 0`
    /// * `align_of::<PhantomCovariantLifetime<'a>>() == 1`
    #[rustc_pub_transparent]
    #[repr(transparent)]
    pub struct PhantomCovariantLifetime<'a>(PhantomCovariant<&'a ()>);
    /// Zero-sized type used to mark a lifetime as contravariant.
    ///
    /// Contravariant lifetimes must live at most as long as declared. See [the reference][1] for
    /// more information.
    ///
    /// [1]: https://doc.rust-lang.org/stable/reference/subtyping.html#variance
    ///
    /// Note: If `'a` is otherwise covariant or invariant, the resulting type is invariant.
    ///
    /// ## Layout
    ///
    /// For all `'a`, the following are guaranteed:
    /// * `size_of::<PhantomContravariantLifetime<'a>>() == 0`
    /// * `align_of::<PhantomContravariantLifetime<'a>>() == 1`
    #[rustc_pub_transparent]
    #[repr(transparent)]
    pub struct PhantomContravariantLifetime<'a>(PhantomContravariant<&'a ()>);
    /// Zero-sized type used to mark a lifetime as invariant.
    ///
    /// Invariant lifetimes must be live for the exact length declared, neither shorter nor longer.
    /// See [the reference][1] for more information.
    ///
    /// [1]: https://doc.rust-lang.org/stable/reference/subtyping.html#variance
    ///
    /// ## Layout
    ///
    /// For all `'a`, the following are guaranteed:
    /// * `size_of::<PhantomInvariantLifetime<'a>>() == 0`
    /// * `align_of::<PhantomInvariantLifetime<'a>>() == 1`
    #[rustc_pub_transparent]
    #[repr(transparent)]
    pub struct PhantomInvariantLifetime<'a>(PhantomInvariant<&'a ()>);
}

phantom_type! {
    /// Zero-sized type used to mark a type parameter as covariant.
    ///
    /// Types used as part of the return value from a function are covariant. If the type is _also_
    /// passed as a parameter then it is [invariant][PhantomInvariant]. See [the reference][1] for
    /// more information.
    ///
    /// [1]: https://doc.rust-lang.org/stable/reference/subtyping.html#variance
    ///
    /// Note: If `T` is otherwise contravariant or invariant, the resulting type is invariant.
    ///
    /// ## Layout
    ///
    /// For all `T`, the following are guaranteed:
    /// * `size_of::<PhantomCovariant<T>>() == 0`
    /// * `align_of::<PhantomCovariant<T>>() == 1`
    #[rustc_pub_transparent]
    #[repr(transparent)]
    pub struct PhantomCovariant<T>(PhantomData<fn() -> T>);
    /// Zero-sized type used to mark a type parameter as contravariant.
    ///
    /// Types passed as arguments to a function are contravariant. If the type is _also_ part of the
    /// return value from a function then it is [invariant][PhantomInvariant]. See [the
    /// reference][1] for more information.
    ///
    /// [1]: https://doc.rust-lang.org/stable/reference/subtyping.html#variance
    ///
    /// Note: If `T` is otherwise covariant or invariant, the resulting type is invariant.
    ///
    /// ## Layout
    ///
    /// For all `T`, the following are guaranteed:
    /// * `size_of::<PhantomContravariant<T>>() == 0`
    /// * `align_of::<PhantomContravariant<T>>() == 1`
    #[rustc_pub_transparent]
    #[repr(transparent)]
    pub struct PhantomContravariant<T>(PhantomData<fn(T)>);
    /// Zero-sized type used to mark a type parameter as invariant.
    ///
    /// Types that are both passed as an argument _and_ used as part of the return value from a
    /// function are invariant. See [the reference][1] for more information.
    ///
    /// [1]: https://doc.rust-lang.org/stable/reference/subtyping.html#variance
    ///
    /// ## Layout
    ///
    /// For all `T`, the following are guaranteed:
    /// * `size_of::<PhantomInvariant<T>>() == 0`
    /// * `align_of::<PhantomInvariant<T>>() == 1`
    #[rustc_pub_transparent]
    #[repr(transparent)]
    pub struct PhantomInvariant<T>(PhantomData<fn(T) -> T>);
}

mod sealed {
    pub trait Sealed {
        const VALUE: Self;
    }
}

/// A marker trait for phantom variance types.
pub trait Variance: sealed::Sealed + Default {}

/// Construct a variance marker; equivalent to [`Default::default`].
///
/// This type can be any of the following. You generally should not need to explicitly name the
/// type, however.
///
/// - [`PhantomCovariant`]
/// - [`PhantomContravariant`]
/// - [`PhantomInvariant`]
/// - [`PhantomCovariantLifetime`]
/// - [`PhantomContravariantLifetime`]
/// - [`PhantomInvariantLifetime`]
///
/// # Example
///
/// ```rust
/// #![feature(phantom_variance_markers)]
///
/// use core::marker::{PhantomCovariant, variance};
///
/// struct BoundFn<F, P, R>
/// where
///     F: Fn(P) -> R,
/// {
///     function: F,
///     parameter: P,
///     return_value: PhantomCovariant<R>,
/// }
///
/// let bound_fn = BoundFn {
///     function: core::convert::identity,
///     parameter: 5u8,
///     return_value: variance(),
/// };
/// ```
pub const fn variance<T>() -> T
where
    T: Variance,
{
    T::VALUE
}
