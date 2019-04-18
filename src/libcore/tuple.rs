// See src/libstd/primitive_docs.rs for documentation.

use crate::cmp::*;
use crate::cmp::Ordering::*;

// macro for implementing n-ary tuple functions and operations
macro_rules! tuple_impls {
    ($(
        $Tuple:ident {
            $(($idx:tt) -> $T:ident)+
        }
    )+) => {
        $(
            #[stable(feature = "rust1", since = "1.0.0")]
            impl<$($T:PartialEq),+> PartialEq for ($($T,)+) where last_type!($($T,)+): ?Sized {
                #[inline]
                fn eq(&self, other: &($($T,)+)) -> bool {
                    $(self.$idx == other.$idx)&&+
                }
                #[inline]
                fn ne(&self, other: &($($T,)+)) -> bool {
                    $(self.$idx != other.$idx)||+
                }
            }

            #[stable(feature = "rust1", since = "1.0.0")]
            impl<$($T:Eq),+> Eq for ($($T,)+) where last_type!($($T,)+): ?Sized {}

            #[stable(feature = "rust1", since = "1.0.0")]
            impl<$($T:PartialOrd + PartialEq),+> PartialOrd for ($($T,)+)
                    where last_type!($($T,)+): ?Sized {
                #[inline]
                fn partial_cmp(&self, other: &($($T,)+)) -> Option<Ordering> {
                    lexical_partial_cmp!($(self.$idx, other.$idx),+)
                }
                #[inline]
                fn lt(&self, other: &($($T,)+)) -> bool {
                    lexical_ord!(lt, $(self.$idx, other.$idx),+)
                }
                #[inline]
                fn le(&self, other: &($($T,)+)) -> bool {
                    lexical_ord!(le, $(self.$idx, other.$idx),+)
                }
                #[inline]
                fn ge(&self, other: &($($T,)+)) -> bool {
                    lexical_ord!(ge, $(self.$idx, other.$idx),+)
                }
                #[inline]
                fn gt(&self, other: &($($T,)+)) -> bool {
                    lexical_ord!(gt, $(self.$idx, other.$idx),+)
                }
            }

            #[stable(feature = "rust1", since = "1.0.0")]
            impl<$($T:Ord),+> Ord for ($($T,)+) where last_type!($($T,)+): ?Sized {
                #[inline]
                fn cmp(&self, other: &($($T,)+)) -> Ordering {
                    lexical_cmp!($(self.$idx, other.$idx),+)
                }
            }

            #[stable(feature = "rust1", since = "1.0.0")]
            impl<$($T:Default),+> Default for ($($T,)+) {
                #[inline]
                fn default() -> ($($T,)+) {
                    ($({ let x: $T = Default::default(); x},)+)
                }
            }
        )+
    }
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

tuple_impls! {
    Tuple1 {
        (0) -> A
    }
    Tuple2 {
        (0) -> A
        (1) -> B
    }
    Tuple3 {
        (0) -> A
        (1) -> B
        (2) -> C
    }
    Tuple4 {
        (0) -> A
        (1) -> B
        (2) -> C
        (3) -> D
    }
    Tuple5 {
        (0) -> A
        (1) -> B
        (2) -> C
        (3) -> D
        (4) -> E
    }
    Tuple6 {
        (0) -> A
        (1) -> B
        (2) -> C
        (3) -> D
        (4) -> E
        (5) -> F
    }
    Tuple7 {
        (0) -> A
        (1) -> B
        (2) -> C
        (3) -> D
        (4) -> E
        (5) -> F
        (6) -> G
    }
    Tuple8 {
        (0) -> A
        (1) -> B
        (2) -> C
        (3) -> D
        (4) -> E
        (5) -> F
        (6) -> G
        (7) -> H
    }
    Tuple9 {
        (0) -> A
        (1) -> B
        (2) -> C
        (3) -> D
        (4) -> E
        (5) -> F
        (6) -> G
        (7) -> H
        (8) -> I
    }
    Tuple10 {
        (0) -> A
        (1) -> B
        (2) -> C
        (3) -> D
        (4) -> E
        (5) -> F
        (6) -> G
        (7) -> H
        (8) -> I
        (9) -> J
    }
    Tuple11 {
        (0) -> A
        (1) -> B
        (2) -> C
        (3) -> D
        (4) -> E
        (5) -> F
        (6) -> G
        (7) -> H
        (8) -> I
        (9) -> J
        (10) -> K
    }
    Tuple12 {
        (0) -> A
        (1) -> B
        (2) -> C
        (3) -> D
        (4) -> E
        (5) -> F
        (6) -> G
        (7) -> H
        (8) -> I
        (9) -> J
        (10) -> K
        (11) -> L
    }
}
