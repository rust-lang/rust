#![feature(decl_macro)]

#[rustfmt::skip]
macro x($macro_name:ident, $macro2_name:ident, $type_name:ident, $variant_name:ident) {
    #[repr(u8)]
    pub enum $type_name {
        Variant = 0,
        $variant_name = 1,
    }

    #[macro_export]
    macro_rules! $macro_name {
        () => {{
            assert_eq!($type_name::Variant as u8, 0);
            assert_eq!($type_name::$variant_name as u8, 1);
            assert_eq!(<$type_name>::Variant as u8, 0);
            assert_eq!(<$type_name>::$variant_name as u8, 1);
        }};
    }

    pub macro $macro2_name {
        () => {{
            assert_eq!($type_name::Variant as u8, 0);
            assert_eq!($type_name::$variant_name as u8, 1);
            assert_eq!(<$type_name>::Variant as u8, 0);
            assert_eq!(<$type_name>::$variant_name as u8, 1);
        }},
    }
}

x!(test_variants, test_variants2, MyEnum, Variant);

pub fn check_variants() {
    test_variants!();
    test_variants2!();
}
