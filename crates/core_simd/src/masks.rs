macro_rules! define_mask {
    { $(#[$attr:meta])* struct $name:ident($type:ty); } => {
        $(#[$attr])*
        #[allow(non_camel_case_types)]
        #[derive(Copy, Clone, Default, PartialEq, PartialOrd, Eq, Ord, Hash)]
        #[repr(transparent)]
        pub struct $name(pub(crate) $type);

        impl $name {
            /// Construct a mask from the given value.
            pub const fn new(value: bool) -> Self {
                if value {
                    Self(!0)
                } else {
                    Self(0)
                }
            }

            /// Test if the mask is set.
            pub const fn test(&self) -> bool {
                self.0 != 0
            }
        }

        impl core::convert::From<bool> for $name {
            fn from(value: bool) -> Self {
                Self::new(value)
            }
        }

        impl core::convert::From<$name> for bool {
            fn from(mask: $name) -> Self {
                mask.test()
            }
        }

        impl core::fmt::Debug for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                self.test().fmt(f)
            }
        }
    }
}

define_mask! {
    #[doc = "8-bit mask"]
    struct mask8(i8);
}

define_mask! {
    #[doc = "16-bit mask"]
    struct mask16(i16);
}

define_mask! {
    #[doc = "32-bit mask"]
    struct mask32(i32);
}

define_mask! {
    #[doc = "64-bit mask"]
    struct mask64(i64);
}

define_mask! {
    #[doc = "128-bit mask"]
    struct mask128(i128);
}

define_mask! {
    #[doc = "`isize`-wide mask"]
    struct masksize(isize);
}
