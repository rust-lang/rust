macro_rules! TryFromU32 {
    derive() (
        $(#[$meta:meta])*
        $vis:vis enum $Type:ident {
            $(
                $(#[$varmeta:meta])*
                $Variant:ident $(= $discr:expr)?
            ),* $(,)?
        }
    ) => {
        impl ::core::convert::TryFrom<u32> for $Type {
            type Error = u32;
            #[allow(deprecated)] // Don't warn about deprecated variants.
            fn try_from(value: u32) -> ::core::result::Result<$Type, Self::Error> {
                $( if value == const { $Type::$Variant as u32 } { return Ok($Type::$Variant) } )*
                Err(value)
            }
        }
    }
}

pub(crate) use TryFromU32;
