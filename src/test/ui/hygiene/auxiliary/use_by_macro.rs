#![feature(decl_macro)]

macro x($macro_name:ident) {
    #[macro_export]
    macro_rules! $macro_name {
        (define) => {
            pub struct MyStruct;
        };
        (create) => {
            MyStruct {}
        };
    }
}

x!(my_struct);
