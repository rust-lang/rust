// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#![crate_name = "rustc_bitflags"]
#![feature(associated_consts)]
#![feature(staged_api)]
#![crate_type = "rlib"]
#![no_std]
#![unstable(feature = "rustc_private", issue = "27812")]

//! A typesafe bitmask flag generator.

#[cfg(test)]
#[macro_use]
extern crate std;

/// The `bitflags!` macro generates a `struct` that holds a set of C-style
/// bitmask flags. It is useful for creating typesafe wrappers for C APIs.
///
/// The flags should only be defined for integer types, otherwise unexpected
/// type errors may occur at compile time.
///
/// # Examples
///
/// ```{.rust}
/// #![feature(rustc_private)]
/// #![feature(associated_consts)]
/// #[macro_use] extern crate rustc_bitflags;
///
/// bitflags! {
///     flags Flags: u32 {
///         const FLAG_A       = 0b00000001,
///         const FLAG_B       = 0b00000010,
///         const FLAG_C       = 0b00000100,
///         const FLAG_ABC     = Flags::FLAG_A.bits
///                            | Flags::FLAG_B.bits
///                            | Flags::FLAG_C.bits,
///     }
/// }
///
/// fn main() {
///     let e1 = Flags::FLAG_A | Flags::FLAG_C;
///     let e2 = Flags::FLAG_B | Flags::FLAG_C;
///     assert!((e1 | e2) == Flags::FLAG_ABC); // union
///     assert!((e1 & e2) == Flags::FLAG_C);   // intersection
///     assert!((e1 - e2) == Flags::FLAG_A);   // set difference
///     assert!(!e2 == Flags::FLAG_A);         // set complement
/// }
/// ```
///
/// The generated `struct`s can also be extended with type and trait implementations:
///
/// ```{.rust}
/// #![feature(rustc_private)]
/// #[macro_use] extern crate rustc_bitflags;
///
/// use std::fmt;
///
/// bitflags! {
///     flags Flags: u32 {
///         const FLAG_A   = 0b00000001,
///         const FLAG_B   = 0b00000010,
///     }
/// }
///
/// impl Flags {
///     pub fn clear(&mut self) {
///         self.bits = 0;  // The `bits` field can be accessed from within the
///                         // same module where the `bitflags!` macro was invoked.
///     }
/// }
///
/// impl fmt::Debug for Flags {
///     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
///         write!(f, "hi!")
///     }
/// }
///
/// fn main() {
///     let mut flags = Flags::FLAG_A | Flags::FLAG_B;
///     flags.clear();
///     assert!(flags.is_empty());
///     assert_eq!(format!("{:?}", flags), "hi!");
/// }
/// ```
///
/// # Attributes
///
/// Attributes can be attached to the generated `struct` by placing them
/// before the `flags` keyword.
///
/// # Derived traits
///
/// The `PartialEq` and `Clone` traits are automatically derived for the `struct` using
/// the `deriving` attribute. Additional traits can be derived by providing an
/// explicit `deriving` attribute on `flags`.
///
/// # Operators
///
/// The following operator traits are implemented for the generated `struct`:
///
/// - `BitOr`: union
/// - `BitAnd`: intersection
/// - `BitXor`: toggle
/// - `Sub`: set difference
/// - `Not`: set complement
///
/// # Methods
///
/// The following methods are defined for the generated `struct`:
///
/// - `empty`: an empty set of flags
/// - `all`: the set of all flags
/// - `bits`: the raw value of the flags currently stored
/// - `from_bits`: convert from underlying bit representation, unless that
///                representation contains bits that do not correspond to a flag
/// - `from_bits_truncate`: convert from underlying bit representation, dropping
///                         any bits that do not correspond to flags
/// - `is_empty`: `true` if no flags are currently stored
/// - `is_all`: `true` if all flags are currently set
/// - `intersects`: `true` if there are flags common to both `self` and `other`
/// - `contains`: `true` all of the flags in `other` are contained within `self`
/// - `insert`: inserts the specified flags in-place
/// - `remove`: removes the specified flags in-place
/// - `toggle`: the specified flags will be inserted if not present, and removed
///             if they are.
#[macro_export]
macro_rules! bitflags {
    ($(#[$attr:meta])* flags $BitFlags:ident: $T:ty {
        $($(#[$Flag_attr:meta])* const $Flag:ident = $value:expr),+
    }) => {
        #[derive(Copy, PartialEq, Eq, Clone, PartialOrd, Ord, Hash)]
        $(#[$attr])*
        pub struct $BitFlags {
            bits: $T,
        }

        impl $BitFlags {
            $($(#[$Flag_attr])* pub const $Flag: $BitFlags = $BitFlags { bits: $value };)+

            /// Returns an empty set of flags.
            #[inline]
            pub fn empty() -> $BitFlags {
                $BitFlags { bits: 0 }
            }

            /// Returns the set containing all flags.
            #[inline]
            pub fn all() -> $BitFlags {
                $BitFlags { bits: $($value)|+ }
            }

            /// Returns the raw value of the flags currently stored.
            #[inline]
            pub fn bits(&self) -> $T {
                self.bits
            }

            /// Convert from underlying bit representation, unless that
            /// representation contains bits that do not correspond to a flag.
            #[inline]
            pub fn from_bits(bits: $T) -> ::std::option::Option<$BitFlags> {
                if (bits & !$BitFlags::all().bits()) != 0 {
                    ::std::option::Option::None
                } else {
                    ::std::option::Option::Some($BitFlags { bits: bits })
                }
            }

            /// Convert from underlying bit representation, dropping any bits
            /// that do not correspond to flags.
            #[inline]
            pub fn from_bits_truncate(bits: $T) -> $BitFlags {
                $BitFlags { bits: bits } & $BitFlags::all()
            }

            /// Returns `true` if no flags are currently stored.
            #[inline]
            pub fn is_empty(&self) -> bool {
                *self == $BitFlags::empty()
            }

            /// Returns `true` if all flags are currently set.
            #[inline]
            pub fn is_all(&self) -> bool {
                *self == $BitFlags::all()
            }

            /// Returns `true` if there are flags common to both `self` and `other`.
            #[inline]
            pub fn intersects(&self, other: $BitFlags) -> bool {
                !(*self & other).is_empty()
            }

            /// Returns `true` all of the flags in `other` are contained within `self`.
            #[inline]
            pub fn contains(&self, other: $BitFlags) -> bool {
                (*self & other) == other
            }

            /// Inserts the specified flags in-place.
            #[inline]
            pub fn insert(&mut self, other: $BitFlags) {
                self.bits |= other.bits;
            }

            /// Removes the specified flags in-place.
            #[inline]
            pub fn remove(&mut self, other: $BitFlags) {
                self.bits &= !other.bits;
            }

            /// Toggles the specified flags in-place.
            #[inline]
            pub fn toggle(&mut self, other: $BitFlags) {
                self.bits ^= other.bits;
            }
        }

        impl ::std::ops::BitOr for $BitFlags {
            type Output = $BitFlags;

            /// Returns the union of the two sets of flags.
            #[inline]
            fn bitor(self, other: $BitFlags) -> $BitFlags {
                $BitFlags { bits: self.bits | other.bits }
            }
        }

        impl ::std::ops::BitXor for $BitFlags {
            type Output = $BitFlags;

            /// Returns the left flags, but with all the right flags toggled.
            #[inline]
            fn bitxor(self, other: $BitFlags) -> $BitFlags {
                $BitFlags { bits: self.bits ^ other.bits }
            }
        }

        impl ::std::ops::BitAnd for $BitFlags {
            type Output = $BitFlags;

            /// Returns the intersection between the two sets of flags.
            #[inline]
            fn bitand(self, other: $BitFlags) -> $BitFlags {
                $BitFlags { bits: self.bits & other.bits }
            }
        }

        impl ::std::ops::Sub for $BitFlags {
            type Output = $BitFlags;

            /// Returns the set difference of the two sets of flags.
            #[inline]
            fn sub(self, other: $BitFlags) -> $BitFlags {
                $BitFlags { bits: self.bits & !other.bits }
            }
        }

        impl ::std::ops::Not for $BitFlags {
            type Output = $BitFlags;

            /// Returns the complement of this set of flags.
            #[inline]
            fn not(self) -> $BitFlags {
                $BitFlags { bits: !self.bits } & $BitFlags::all()
            }
        }
    };
    ($(#[$attr:meta])* flags $BitFlags:ident: $T:ty {
        $($(#[$Flag_attr:meta])* const $Flag:ident = $value:expr),+,
    }) => {
        bitflags! {
            $(#[$attr])*
            flags $BitFlags: $T {
                $($(#[$Flag_attr])* const $Flag = $value),+
            }
        }
    };
}

#[cfg(test)]
#[allow(non_upper_case_globals)]
mod tests {
    use std::hash::{Hasher, Hash, SipHasher};
    use std::option::Option::{Some, None};

    bitflags! {
        #[doc = "> The first principle is that you must not fool yourself — and"]
        #[doc = "> you are the easiest person to fool."]
        #[doc = "> "]
        #[doc = "> - Richard Feynman"]
        flags Flags: u32 {
            const FlagA       = 0b00000001,
            #[doc = "<pcwalton> macros are way better at generating code than trans is"]
            const FlagB       = 0b00000010,
            const FlagC       = 0b00000100,
            #[doc = "* cmr bed"]
            #[doc = "* strcat table"]
            #[doc = "<strcat> wait what?"]
            const FlagABC     = Flags::FlagA.bits
                               | Flags::FlagB.bits
                               | Flags::FlagC.bits,
        }
    }

    bitflags! {
        flags AnotherSetOfFlags: i8 {
            const AnotherFlag = -1,
        }
    }

    #[test]
    fn test_bits() {
        assert_eq!(Flags::empty().bits(), 0b00000000);
        assert_eq!(Flags::FlagA.bits(), 0b00000001);
        assert_eq!(Flags::FlagABC.bits(), 0b00000111);

        assert_eq!(AnotherSetOfFlags::empty().bits(), 0b00);
        assert_eq!(AnotherSetOfFlags::AnotherFlag.bits(), !0);
    }

    #[test]
    fn test_from_bits() {
        assert!(Flags::from_bits(0) == Some(Flags::empty()));
        assert!(Flags::from_bits(0b1) == Some(Flags::FlagA));
        assert!(Flags::from_bits(0b10) == Some(Flags::FlagB));
        assert!(Flags::from_bits(0b11) == Some(Flags::FlagA | Flags::FlagB));
        assert!(Flags::from_bits(0b1000) == None);

        assert!(AnotherSetOfFlags::from_bits(!0) == Some(AnotherSetOfFlags::AnotherFlag));
    }

    #[test]
    fn test_from_bits_truncate() {
        assert!(Flags::from_bits_truncate(0) == Flags::empty());
        assert!(Flags::from_bits_truncate(0b1) == Flags::FlagA);
        assert!(Flags::from_bits_truncate(0b10) == Flags::FlagB);
        assert!(Flags::from_bits_truncate(0b11) == (Flags::FlagA | Flags::FlagB));
        assert!(Flags::from_bits_truncate(0b1000) == Flags::empty());
        assert!(Flags::from_bits_truncate(0b1001) == Flags::FlagA);

        assert!(AnotherSetOfFlags::from_bits_truncate(0) == AnotherSetOfFlags::empty());
    }

    #[test]
    fn test_is_empty() {
        assert!(Flags::empty().is_empty());
        assert!(!Flags::FlagA.is_empty());
        assert!(!Flags::FlagABC.is_empty());

        assert!(!AnotherSetOfFlags::AnotherFlag.is_empty());
    }

    #[test]
    fn test_is_all() {
        assert!(Flags::all().is_all());
        assert!(!Flags::FlagA.is_all());
        assert!(Flags::FlagABC.is_all());

        assert!(AnotherSetOfFlags::AnotherFlag.is_all());
    }

    #[test]
    fn test_two_empties_do_not_intersect() {
        let e1 = Flags::empty();
        let e2 = Flags::empty();
        assert!(!e1.intersects(e2));

        assert!(AnotherSetOfFlags::AnotherFlag.intersects(AnotherSetOfFlags::AnotherFlag));
    }

    #[test]
    fn test_empty_does_not_intersect_with_full() {
        let e1 = Flags::empty();
        let e2 = Flags::FlagABC;
        assert!(!e1.intersects(e2));
    }

    #[test]
    fn test_disjoint_intersects() {
        let e1 = Flags::FlagA;
        let e2 = Flags::FlagB;
        assert!(!e1.intersects(e2));
    }

    #[test]
    fn test_overlapping_intersects() {
        let e1 = Flags::FlagA;
        let e2 = Flags::FlagA | Flags::FlagB;
        assert!(e1.intersects(e2));
    }

    #[test]
    fn test_contains() {
        let e1 = Flags::FlagA;
        let e2 = Flags::FlagA | Flags::FlagB;
        assert!(!e1.contains(e2));
        assert!(e2.contains(e1));
        assert!(Flags::FlagABC.contains(e2));

        assert!(AnotherSetOfFlags::AnotherFlag.contains(AnotherSetOfFlags::AnotherFlag));
    }

    #[test]
    fn test_insert() {
        let mut e1 = Flags::FlagA;
        let e2 = Flags::FlagA | Flags::FlagB;
        e1.insert(e2);
        assert!(e1 == e2);

        let mut e3 = AnotherSetOfFlags::empty();
        e3.insert(AnotherSetOfFlags::AnotherFlag);
        assert!(e3 == AnotherSetOfFlags::AnotherFlag);
    }

    #[test]
    fn test_remove() {
        let mut e1 = Flags::FlagA | Flags::FlagB;
        let e2 = Flags::FlagA | Flags::FlagC;
        e1.remove(e2);
        assert!(e1 == Flags::FlagB);

        let mut e3 = AnotherSetOfFlags::AnotherFlag;
        e3.remove(AnotherSetOfFlags::AnotherFlag);
        assert!(e3 == AnotherSetOfFlags::empty());
    }

    #[test]
    fn test_operators() {
        let e1 = Flags::FlagA | Flags::FlagC;
        let e2 = Flags::FlagB | Flags::FlagC;
        assert!((e1 | e2) == Flags::FlagABC);     // union
        assert!((e1 & e2) == Flags::FlagC);       // intersection
        assert!((e1 - e2) == Flags::FlagA);       // set difference
        assert!(!e2 == Flags::FlagA);             // set complement
        assert!(e1 ^ e2 == Flags::FlagA | Flags::FlagB); // toggle
        let mut e3 = e1;
        e3.toggle(e2);
        assert!(e3 == Flags::FlagA | Flags::FlagB);

        let mut m4 = AnotherSetOfFlags::empty();
        m4.toggle(AnotherSetOfFlags::empty());
        assert!(m4 == AnotherSetOfFlags::empty());
    }

    #[test]
    fn test_lt() {
        let mut a = Flags::empty();
        let mut b = Flags::empty();

        assert!(!(a < b) && !(b < a));
        b = Flags::FlagB;
        assert!(a < b);
        a = Flags::FlagC;
        assert!(!(a < b) && b < a);
        b = Flags::FlagC | Flags::FlagB;
        assert!(a < b);
    }

    #[test]
    fn test_ord() {
        let mut a = Flags::empty();
        let mut b = Flags::empty();

        assert!(a <= b && a >= b);
        a = Flags::FlagA;
        assert!(a > b && a >= b);
        assert!(b < a && b <= a);
        b = Flags::FlagB;
        assert!(b > a && b >= a);
        assert!(a < b && a <= b);
    }

    #[test]
    fn test_hash() {
        let mut x = Flags::empty();
        let mut y = Flags::empty();
        assert!(hash(&x) == hash(&y));
        x = Flags::all();
        y = Flags::FlagABC;
        assert!(hash(&x) == hash(&y));
    }

    fn hash<T: Hash>(t: &T) -> u64 {
        let mut s = SipHasher::new();
        t.hash(&mut s);
        s.finish()
    }
}
