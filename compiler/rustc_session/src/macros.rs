/// Derivable trait for enums with no fields (i.e. C-style enums) that want to
/// allow iteration over a list of all variant values.
pub(crate) trait AllVariants: Copy + 'static {
    const ALL_VARIANTS: &[Self];
}

macro_rules! AllVariantsDerive {
    derive() (
        $(#[$meta:meta])*
        $vis:vis enum $Type:ident {
            $(
                $(#[$varmeta:meta])*
                $Variant:ident $( = $value:literal )?
            ), *$(,)?
        }
    ) => {
        impl $crate::macros::AllVariants for $Type {
            const ALL_VARIANTS: &[$Type] = &[
                $( $Type::$Variant, )*
            ];
        }
    };
}

// For some reason the compiler won't allow `pub(crate) use AllVariants` due
// to a conflict with the trait of the same name, but will allow this form.
pub(crate) use AllVariantsDerive as AllVariants;
