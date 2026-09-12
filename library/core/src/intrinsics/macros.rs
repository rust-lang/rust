//! Macros used to declare intrinsics.

/// Declares an intrinsic that is generic over its type, but whose fallback body has to be
/// written per-type. This is needed for the float intrinsics, which are generics but may
/// have different fallback bodies for each float width.
///
/// ```ignore (illustrative)
/// intrinsic_dispatch_on_type! {
///     /// Returns the exponential of a float.
///     #[rustc_nounwind]
///     #[rustc_intrinsic]
///     pub fn expf<T: bounds::FloatPrimitive>(x: T) -> T;
///
///     f16 => { expf(x as f32) as f16 }
///     f32 => { libm::likely_available::expf(x) }
///     f64 => { libm::likely_available::exp(x) }
///     f128 => { libm::maybe_available::expf128(x) }
/// }
/// ```
macro_rules! intrinsic_dispatch_on_type {
    (
        $(#[$attr:meta])*
        $vis:vis fn $name:ident<$generic:ident: $bound:path>(
            $($arg:ident: $arg_ty:ty),* $(,)?
        ) -> $ret_ty:ty;
        $($concrete:ty => $body:block)*
    ) => {
        mod $name {
            use super::*;

            pub trait Dispatch<$generic = Self>: $bound {
                fn dispatch($($arg: $arg_ty),*) -> $ret_ty;
            }

            intrinsic_dispatch_on_type! {
                @impls $generic, ($($arg: $arg_ty),*) -> $ret_ty,
                $($concrete => $body)*
            }
        }

        $(#[$attr])*
        $vis fn $name<$generic: $name::Dispatch>($($arg: $arg_ty),*) -> $ret_ty {
            <$generic as $name::Dispatch>::dispatch($($arg),*)
        }
    };

    (
        @impls $generic:ident, $args:tt -> $ret_ty:ty,
        $($concrete:ty => $body:block)*
    ) => {
        $(intrinsic_dispatch_on_type! { @impl $generic = $concrete, $args -> $ret_ty $body })*
    };

    (
        @impl $generic:ident = $concrete:ty,
        ($($arg:ident: $arg_ty:ty),*) -> $ret_ty:ty $body:block
    ) => {
        // We need a const block here to define a type alias for the type var ($generic),
        // because the arguments in `fn dispatch` are the ones used to define the intrinsic.
        //
        // For instance, when defining `fn exp<T: bounds::FloatPrimitive>(x: T) -> T`,
        // the definition of `dispatch` below is going to be `fn dispatch(x: T) -> T`.
        const _: () = {
            type $generic = $concrete;

            impl Dispatch<$concrete> for $concrete {
                #[inline]
                fn dispatch($($arg: $arg_ty),*) -> $ret_ty $body
            }
        };
    };
}

pub(super) use intrinsic_dispatch_on_type;
