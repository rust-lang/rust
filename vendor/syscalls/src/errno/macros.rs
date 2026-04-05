// Helper for generating the Errno implementation.
macro_rules! errno_enum {
    (
        $(#[$meta:meta])*
        $vis:vis enum $Name:ident {
            $(
                $(#[$attrs:meta])*
                $item:ident($code:expr) = $doc:expr,
            )*
        }
    ) => {
        $(#[$meta])*
        #[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
        #[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
        $vis struct $Name(pub(super) i32);

        impl $Name {
            $(
                #[doc = $doc]
                $(#[$attrs])*
                pub const $item: $Name = $Name($code);
            )*

            /// Returns a pair containing the name of the error and a string
            /// describing the error.
            pub fn name_and_description(&self) -> Option<(&'static str, &'static str)> {
                match *self {
                    $(
                        $(#[$attrs])*
                        $Name::$item => Some((stringify!($item), $doc)),
                    )*
                    _ => None,
                }
            }
        }
    }
}
