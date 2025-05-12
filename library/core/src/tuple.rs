// See core/src/primitive_docs.rs for documentation.

use crate::cmp::Ordering::{self, *};
use crate::marker::{ConstParamTy_, StructuralPartialEq, UnsizedConstParamTy};
use crate::ops::ControlFlow::{self, Break, Continue};

// Recursive macro for implementing n-ary tuple functions and operations
//
// Also provides implementations for tuples with lesser arity. For example, tuple_impls!(A B C)
// will implement everything for (A, B, C), (A, B) and (A,).
macro_rules! tuple_impls {
    // Stopping criteria (1-ary tuple)
    ($T:ident) => {
        tuple_impls!(@impl $T);
    };
    // Running criteria (n-ary tuple, with n >= 2)
    ($T:ident $( $U:ident )+) => {
        tuple_impls!($( $U )+);
        tuple_impls!(@impl $T $( $U )+);
    };
    // "Private" internal implementation
    (@impl $( $T:ident )+) => {
        maybe_tuple_doc! {
            $($T)+ @
            #[stable(feature = "rust1", since = "1.0.0")]
            impl<$($T: PartialEq),+> PartialEq for ($($T,)+)
            where
                last_type!($($T,)+): ?Sized
            {
                #[inline]
                fn eq(&self, other: &($($T,)+)) -> bool {
                    $( ${ignore($T)} self.${index()} == other.${index()} )&&+
                }
                #[inline]
                fn ne(&self, other: &($($T,)+)) -> bool {
                    $( ${ignore($T)} self.${index()} != other.${index()} )||+
                }
            }
        }

        maybe_tuple_doc! {
            $($T)+ @
            #[stable(feature = "rust1", since = "1.0.0")]
            impl<$($T: Eq),+> Eq for ($($T,)+)
            where
                last_type!($($T,)+): ?Sized
            {}
        }

        maybe_tuple_doc! {
            $($T)+ @
            #[unstable(feature = "adt_const_params", issue = "95174")]
            impl<$($T: ConstParamTy_),+> ConstParamTy_ for ($($T,)+)
            {}
        }

        maybe_tuple_doc! {
            $($T)+ @
            #[unstable(feature = "unsized_const_params", issue = "95174")]
            impl<$($T: UnsizedConstParamTy),+> UnsizedConstParamTy for ($($T,)+)
            {}
        }

        maybe_tuple_doc! {
            $($T)+ @
            #[unstable(feature = "structural_match", issue = "31434")]
            impl<$($T),+> StructuralPartialEq for ($($T,)+)
            {}
        }

        maybe_tuple_doc! {
            $($T)+ @
            #[stable(feature = "rust1", since = "1.0.0")]
            impl<$($T: PartialOrd),+> PartialOrd for ($($T,)+)
            where
                last_type!($($T,)+): ?Sized
            {
                #[inline]
                fn partial_cmp(&self, other: &($($T,)+)) -> Option<Ordering> {
                    lexical_partial_cmp!($( ${ignore($T)} self.${index()}, other.${index()} ),+)
                }
                #[inline]
                fn lt(&self, other: &($($T,)+)) -> bool {
                    lexical_ord!(lt, __chaining_lt, $( ${ignore($T)} self.${index()}, other.${index()} ),+)
                }
                #[inline]
                fn le(&self, other: &($($T,)+)) -> bool {
                    lexical_ord!(le, __chaining_le, $( ${ignore($T)} self.${index()}, other.${index()} ),+)
                }
                #[inline]
                fn ge(&self, other: &($($T,)+)) -> bool {
                    lexical_ord!(ge, __chaining_ge, $( ${ignore($T)} self.${index()}, other.${index()} ),+)
                }
                #[inline]
                fn gt(&self, other: &($($T,)+)) -> bool {
                    lexical_ord!(gt, __chaining_gt, $( ${ignore($T)} self.${index()}, other.${index()} ),+)
                }
                #[inline]
                fn __chaining_lt(&self, other: &($($T,)+)) -> ControlFlow<bool> {
                    lexical_chain!(__chaining_lt, $( ${ignore($T)} self.${index()}, other.${index()} ),+)
                }
                #[inline]
                fn __chaining_le(&self, other: &($($T,)+)) -> ControlFlow<bool> {
                    lexical_chain!(__chaining_le, $( ${ignore($T)} self.${index()}, other.${index()} ),+)
                }
                #[inline]
                fn __chaining_gt(&self, other: &($($T,)+)) -> ControlFlow<bool> {
                    lexical_chain!(__chaining_gt, $( ${ignore($T)} self.${index()}, other.${index()} ),+)
                }
                #[inline]
                fn __chaining_ge(&self, other: &($($T,)+)) -> ControlFlow<bool> {
                    lexical_chain!(__chaining_ge, $( ${ignore($T)} self.${index()}, other.${index()} ),+)
                }
            }
        }

        maybe_tuple_doc! {
            $($T)+ @
            #[stable(feature = "rust1", since = "1.0.0")]
            impl<$($T: Ord),+> Ord for ($($T,)+)
            where
                last_type!($($T,)+): ?Sized
            {
                #[inline]
                fn cmp(&self, other: &($($T,)+)) -> Ordering {
                    lexical_cmp!($( ${ignore($T)} self.${index()}, other.${index()} ),+)
                }
            }
        }

        maybe_tuple_doc! {
            $($T)+ @
            #[stable(feature = "rust1", since = "1.0.0")]
            impl<$($T: Default),+> Default for ($($T,)+) {
                #[inline]
                fn default() -> ($($T,)+) {
                    ($({ let x: $T = Default::default(); x},)+)
                }
            }
        }

        maybe_tuple_doc! {
            $($T)+ @
            #[stable(feature = "array_tuple_conv", since = "1.71.0")]
            impl<T> From<[T; ${count($T)}]> for ($(${ignore($T)} T,)+) {
                #[inline]
                #[allow(non_snake_case)]
                fn from(array: [T; ${count($T)}]) -> Self {
                    let [$($T,)+] = array;
                    ($($T,)+)
                }
            }
        }

        maybe_tuple_doc! {
            $($T)+ @
            #[stable(feature = "array_tuple_conv", since = "1.71.0")]
            impl<T> From<($(${ignore($T)} T,)+)> for [T; ${count($T)}] {
                #[inline]
                #[allow(non_snake_case)]
                fn from(tuple: ($(${ignore($T)} T,)+)) -> Self {
                    let ($($T,)+) = tuple;
                    [$($T,)+]
                }
            }
        }
    }
}

// If this is a unary tuple, it adds a doc comment.
// Otherwise, it hides the docs entirely.
macro_rules! maybe_tuple_doc {
    ($a:ident @ #[$meta:meta] $item:item) => {
        #[doc(fake_variadic)]
        #[doc = "This trait is implemented for tuples up to twelve items long."]
        #[$meta]
        $item
    };
    ($a:ident $($rest_a:ident)+ @ #[$meta:meta] $item:item) => {
        #[doc(hidden)]
        #[$meta]
        $item
    };
}

// Constructs an expression that performs a lexical ordering using method `$rel`.
// The values are interleaved, so the macro invocation for
// `(a1, a2, a3) < (b1, b2, b3)` would be `lexical_ord!(lt, opt_is_lt, a1, b1,
// a2, b2, a3, b3)` (and similarly for `lexical_cmp`)
//
// `$chain_rel` is the chaining method from `PartialOrd` to use for all but the
// final value, to produce better results for simple primitives.
macro_rules! lexical_ord {
    ($rel: ident, $chain_rel: ident, $a:expr, $b:expr, $($rest_a:expr, $rest_b:expr),+) => {{
        match PartialOrd::$chain_rel(&$a, &$b) {
            Break(val) => val,
            Continue(()) => lexical_ord!($rel, $chain_rel, $($rest_a, $rest_b),+),
        }
    }};
    ($rel: ident, $chain_rel: ident, $a:expr, $b:expr) => {
        // Use the specific method for the last element
        PartialOrd::$rel(&$a, &$b)
    };
}

// Same parameter interleaving as `lexical_ord` above
macro_rules! lexical_chain {
    ($chain_rel: ident, $a:expr, $b:expr $(,$rest_a:expr, $rest_b:expr)*) => {{
        PartialOrd::$chain_rel(&$a, &$b)?;
        lexical_chain!($chain_rel $(,$rest_a, $rest_b)*)
    }};
    ($chain_rel: ident) => {
        Continue(())
    };
}

macro_rules! lexical_partial_cmp {
    ($a:expr, $b:expr, $($rest_a:expr, $rest_b:expr),+) => {
        match ($a).partial_cmp(&$b) {
            Some(Equal) => lexical_partial_cmp!($($rest_a, $rest_b),+),
            ordering => ordering
        }
    };
    ($a:expr, $b:expr) => { ($a).partial_cmp(&$b) };
}

macro_rules! lexical_cmp {
    ($a:expr, $b:expr, $($rest_a:expr, $rest_b:expr),+) => {
        match ($a).cmp(&$b) {
            Equal => lexical_cmp!($($rest_a, $rest_b),+),
            ordering => ordering
        }
    };
    ($a:expr, $b:expr) => { ($a).cmp(&$b) };
}

macro_rules! last_type {
    ($a:ident,) => { $a };
    ($a:ident, $($rest_a:ident,)+) => { last_type!($($rest_a,)+) };
}

tuple_impls!(E D C B A Z Y X W V U T);
