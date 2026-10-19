#![unstable(feature = "phantom_variance_markers", issue = "135806")]

use crate::any::type_name;
use crate::clone::TrivialClone;
use crate::cmp::Ordering;
use crate::fmt;
use crate::hash::{Hash, Hasher};

macro_rules! phantom_type {
    ($(
        $(#[$attr:meta])*
        pub struct $name:ident <$(#[$param_attr:meta])? $t:ident>;
    )*) => {$(
        $(#[$attr])*
        pub struct $name<$(#[$param_attr])? $t: ?Sized>;

        #[rustc_const_unstable(feature = "const_default", issue = "143894")]
        const impl<T> Default for $name<T>
            where T: ?Sized
        {
            fn default() -> Self {
                Self
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

        #[doc(hidden)]
        unsafe impl<T> TrivialClone for $name<T> where T: ?Sized {}

        impl<T> PartialEq for $name<T>
            where T: ?Sized
        {
            fn eq(&self, _: &Self) -> bool {
                true
            }
        }

        impl<T> Eq for $name<T> where T: ?Sized {}

        impl<T> super::StructuralPartialEq for $name<T> where T: ?Sized {}

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
        pub struct $name:ident <$(#[$param_attr:meta])? $lt:lifetime>;
    )*) => {$(
        $(#[$attr])*
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name<$(#[$param_attr])? $lt>;

        #[rustc_const_unstable(feature = "const_default", issue = "143894")]
        const impl Default for $name<'_> {
            fn default() -> Self {
                Self
            }
        }

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
    #[rustc_phantom_marker]
    pub struct PhantomCovariantLifetime<#[rustc_force_variance(covariant)] 'a>;
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
    #[rustc_phantom_marker]
    pub struct PhantomContravariantLifetime<#[rustc_force_variance(contravariant)] 'a>;
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
    #[rustc_phantom_marker]
    pub struct PhantomInvariantLifetime<#[rustc_force_variance(invariant)] 'a>;
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
    #[rustc_phantom_marker]
    pub struct PhantomCovariant<#[rustc_force_variance(covariant)] T>;
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
    #[rustc_phantom_marker]
    pub struct PhantomContravariant<#[rustc_force_variance(contravariant)] T>;
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
    #[rustc_phantom_marker]
    pub struct PhantomInvariant<#[rustc_force_variance(invariant)] T>;
}
