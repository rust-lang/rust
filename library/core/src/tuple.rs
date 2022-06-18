// See src/libstd/primitive_docs.rs for documentation.

use crate::cmp::Ordering::*;
use crate::cmp::*;

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
            impl<$($T:PartialEq),+> PartialEq for ($($T,)+)
            where
                last_type!($($T,)+): ?Sized
            {
                #[inline]
                fn eq(&self, other: &($($T,)+)) -> bool {
                    $( ${ignore(T)} self.${index()} == other.${index()} )&&+
                }
                #[inline]
                fn ne(&self, other: &($($T,)+)) -> bool {
                    $( ${ignore(T)} self.${index()} != other.${index()} )||+
                }
            }
        }

        maybe_tuple_doc! {
            $($T)+ @
            #[stable(feature = "rust1", since = "1.0.0")]
            impl<$($T:Eq),+> Eq for ($($T,)+)
            where
                last_type!($($T,)+): ?Sized
            {}
        }

        maybe_tuple_doc! {
            $($T)+ @
            #[stable(feature = "rust1", since = "1.0.0")]
            impl<$($T:PartialOrd + PartialEq),+> PartialOrd for ($($T,)+)
            where
                last_type!($($T,)+): ?Sized
            {
                #[inline]
                fn partial_cmp(&self, other: &($($T,)+)) -> Option<Ordering> {
                    lexical_partial_cmp!($( ${ignore(T)} self.${index()}, other.${index()} ),+)
                }
                #[inline]
                fn lt(&self, other: &($($T,)+)) -> bool {
                    lexical_ord!(lt, $( ${ignore(T)} self.${index()}, other.${index()} ),+)
                }
                #[inline]
                fn le(&self, other: &($($T,)+)) -> bool {
                    lexical_ord!(le, $( ${ignore(T)} self.${index()}, other.${index()} ),+)
                }
                #[inline]
                fn ge(&self, other: &($($T,)+)) -> bool {
                    lexical_ord!(ge, $( ${ignore(T)} self.${index()}, other.${index()} ),+)
                }
                #[inline]
                fn gt(&self, other: &($($T,)+)) -> bool {
                    lexical_ord!(gt, $( ${ignore(T)} self.${index()}, other.${index()} ),+)
                }
            }
        }

        maybe_tuple_doc! {
            $($T)+ @
            #[stable(feature = "rust1", since = "1.0.0")]
            impl<$($T:Ord),+> Ord for ($($T,)+)
            where
                last_type!($($T,)+): ?Sized
            {
                #[inline]
                fn cmp(&self, other: &($($T,)+)) -> Ordering {
                    lexical_cmp!($( ${ignore(T)} self.${index()}, other.${index()} ),+)
                }
            }
        }

        maybe_tuple_doc! {
            $($T)+ @
            #[stable(feature = "rust1", since = "1.0.0")]
            impl<$($T:Default),+> Default for ($($T,)+) {
                #[inline]
                fn default() -> ($($T,)+) {
                    ($({ let x: $T = Default::default(); x},)+)
                }
            }
        }
    }
}

// If this is a unary tuple, it adds a doc comment.
// Otherwise, it hides the docs entirely.
macro_rules! maybe_tuple_doc {
    ($a:ident @ #[$meta:meta] $item:item) => {
        #[cfg_attr(not(bootstrap), doc(tuple_variadic))]
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

// Constructs an expression that performs a lexical ordering using method $rel.
// The values are interleaved, so the macro invocation for
// `(a1, a2, a3) < (b1, b2, b3)` would be `lexical_ord!(lt, a1, b1, a2, b2,
// a3, b3)` (and similarly for `lexical_cmp`)
macro_rules! lexical_ord {
    ($rel: ident, $a:expr, $b:expr, $($rest_a:expr, $rest_b:expr),+) => {
        if $a != $b { lexical_ord!($rel, $a, $b) }
        else { lexical_ord!($rel, $($rest_a, $rest_b),+) }
    };
    ($rel: ident, $a:expr, $b:expr) => { ($a) . $rel (& $b) };
}

macro_rules! lexical_partial_cmp {
    ($a:expr, $b:expr, $($rest_a:expr, $rest_b:expr),+) => {
        match ($a).partial_cmp(&$b) {
            Some(Equal) => lexical_partial_cmp!($($rest_a, $rest_b),+),
            ordering   => ordering
        }
    };
    ($a:expr, $b:expr) => { ($a).partial_cmp(&$b) };
}

macro_rules! lexical_cmp {
    ($a:expr, $b:expr, $($rest_a:expr, $rest_b:expr),+) => {
        match ($a).cmp(&$b) {
            Equal => lexical_cmp!($($rest_a, $rest_b),+),
            ordering   => ordering
        }
    };
    ($a:expr, $b:expr) => { ($a).cmp(&$b) };
}

macro_rules! last_type {
    ($a:ident,) => { $a };
    ($a:ident, $($rest_a:ident,)+) => { last_type!($($rest_a,)+) };
}

tuple_impls!(E D C B A Z Y X W V U T);
