#![feature(decl_macro)]

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Field {
    RootCtxt,
    MacroCtxt,
}

#[rustfmt::skip]
macro x(
    $macro_name:ident,
    $macro2_name:ident,
    $type_name:ident,
    $field_name:ident,
    $const_name:ident
) {
    #[derive(Copy, Clone)]
    pub struct $type_name {
        pub field: Field,
        pub $field_name: Field,
    }

    pub const $const_name: $type_name =
        $type_name { field: Field::MacroCtxt, $field_name: Field::RootCtxt };

    #[macro_export]
    macro_rules! $macro_name {
        (check_fields_of $e:expr) => {{
            let e = $e;
            assert_eq!(e.field, Field::MacroCtxt);
            assert_eq!(e.$field_name, Field::RootCtxt);
        }};
        (check_fields) => {{
            assert_eq!($const_name.field, Field::MacroCtxt);
            assert_eq!($const_name.$field_name, Field::RootCtxt);
        }};
        (construct) => {
            $type_name { field: Field::MacroCtxt, $field_name: Field::RootCtxt }
        };
    }

    pub macro $macro2_name {
        (check_fields_of $e:expr) => {{
            let e = $e;
            assert_eq!(e.field, Field::MacroCtxt);
            assert_eq!(e.$field_name, Field::RootCtxt);
        }},
        (check_fields) => {{
            assert_eq!($const_name.field, Field::MacroCtxt);
            assert_eq!($const_name.$field_name, Field::RootCtxt);
        }},
        (construct) => {
            $type_name { field: Field::MacroCtxt, $field_name: Field::RootCtxt }
        }
    }
}

x!(test_fields, test_fields2, MyStruct, field, MY_CONST);

pub fn check_fields(s: MyStruct) {
    test_fields!(check_fields_of s);
}

pub fn check_fields_local() {
    test_fields!(check_fields);
    test_fields2!(check_fields);

    let s1 = test_fields!(construct);
    test_fields!(check_fields_of s1);

    let s2 = test_fields2!(construct);
    test_fields2!(check_fields_of s2);
}
