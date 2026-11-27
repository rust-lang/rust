#![feature(decl_macro)]

#[derive(PartialEq, Eq, Debug)]
pub enum Method {
    DefaultMacroCtxt,
    DefaultRootCtxt,
    OverrideMacroCtxt,
    OverrideRootCtxt,
}

#[rustfmt::skip]
macro x($macro_name:ident, $macro2_name:ident, $trait_name:ident, $method_name:ident) {
    pub trait $trait_name {
        fn method(&self) -> Method {
            Method::DefaultMacroCtxt
        }

        fn $method_name(&self) -> Method {
            Method::DefaultRootCtxt
        }
    }

    impl $trait_name for () {}
    impl $trait_name for bool {
        fn method(&self) -> Method {
            Method::OverrideMacroCtxt
        }

        fn $method_name(&self) -> Method {
            Method::OverrideRootCtxt
        }
    }

    #[macro_export]
    macro_rules! $macro_name {
        (check_resolutions) => {
            assert_eq!(().method(), Method::DefaultMacroCtxt);
            assert_eq!($trait_name::method(&()), Method::DefaultMacroCtxt);
            assert_eq!(().$method_name(), Method::DefaultRootCtxt);
            assert_eq!($trait_name::$method_name(&()), Method::DefaultRootCtxt);

            assert_eq!(false.method(), Method::OverrideMacroCtxt);
            assert_eq!($trait_name::method(&false), Method::OverrideMacroCtxt);
            assert_eq!(false.$method_name(), Method::OverrideRootCtxt);
            assert_eq!($trait_name::$method_name(&false), Method::OverrideRootCtxt);

            assert_eq!('a'.method(), Method::DefaultMacroCtxt);
            assert_eq!($trait_name::method(&'a'), Method::DefaultMacroCtxt);
            assert_eq!('a'.$method_name(), Method::DefaultRootCtxt);
            assert_eq!($trait_name::$method_name(&'a'), Method::DefaultRootCtxt);

            assert_eq!(1i32.method(), Method::OverrideMacroCtxt);
            assert_eq!($trait_name::method(&1i32), Method::OverrideMacroCtxt);
            assert_eq!(1i32.$method_name(), Method::OverrideRootCtxt);
            assert_eq!($trait_name::$method_name(&1i32), Method::OverrideRootCtxt);

            assert_eq!(1i64.method(), Method::OverrideMacroCtxt);
            assert_eq!($trait_name::method(&1i64), Method::OverrideMacroCtxt);
            assert_eq!(1i64.$method_name(), Method::OverrideRootCtxt);
            assert_eq!($trait_name::$method_name(&1i64), Method::OverrideRootCtxt);
        };
        (assert_no_override $v:expr) => {
            assert_eq!($v.method(), Method::DefaultMacroCtxt);
            assert_eq!($trait_name::method(&$v), Method::DefaultMacroCtxt);
            assert_eq!($v.$method_name(), Method::DefaultRootCtxt);
            assert_eq!($trait_name::$method_name(&$v), Method::DefaultRootCtxt);
        };
        (assert_override $v:expr) => {
            assert_eq!($v.method(), Method::OverrideMacroCtxt);
            assert_eq!($trait_name::method(&$v), Method::OverrideMacroCtxt);
            assert_eq!($v.$method_name(), Method::OverrideRootCtxt);
            assert_eq!($trait_name::$method_name(&$v), Method::OverrideRootCtxt);
        };
        (impl for $t:ty) => {
            impl $trait_name for $t {
                fn method(&self) -> Method {
                    Method::OverrideMacroCtxt
                }

                fn $method_name(&self) -> Method {
                    Method::OverrideRootCtxt
                }
            }
        };
    }

    pub macro $macro2_name {
        (check_resolutions) => {
            assert_eq!(().method(), Method::DefaultMacroCtxt);
            assert_eq!($trait_name::method(&()), Method::DefaultMacroCtxt);
            assert_eq!(().$method_name(), Method::DefaultRootCtxt);
            assert_eq!($trait_name::$method_name(&()), Method::DefaultRootCtxt);

            assert_eq!(false.method(), Method::OverrideMacroCtxt);
            assert_eq!($trait_name::method(&false), Method::OverrideMacroCtxt);
            assert_eq!(false.$method_name(), Method::OverrideRootCtxt);
            assert_eq!($trait_name::$method_name(&false), Method::OverrideRootCtxt);

            assert_eq!('a'.method(), Method::DefaultMacroCtxt);
            assert_eq!($trait_name::method(&'a'), Method::DefaultMacroCtxt);
            assert_eq!('a'.$method_name(), Method::DefaultRootCtxt);
            assert_eq!($trait_name::$method_name(&'a'), Method::DefaultRootCtxt);

            assert_eq!(1i32.method(), Method::OverrideMacroCtxt);
            assert_eq!($trait_name::method(&1i32), Method::OverrideMacroCtxt);
            assert_eq!(1i32.$method_name(), Method::OverrideRootCtxt);
            assert_eq!($trait_name::$method_name(&1i32), Method::OverrideRootCtxt);

            assert_eq!(1i64.method(), Method::OverrideMacroCtxt);
            assert_eq!($trait_name::method(&1i64), Method::OverrideMacroCtxt);
            assert_eq!(1i64.$method_name(), Method::OverrideRootCtxt);
            assert_eq!($trait_name::$method_name(&1i64), Method::OverrideRootCtxt);
        },
        (assert_no_override $v:expr) => {
            assert_eq!($v.method(), Method::DefaultMacroCtxt);
            assert_eq!($trait_name::method(&$v), Method::DefaultMacroCtxt);
            assert_eq!($v.$method_name(), Method::DefaultRootCtxt);
            assert_eq!($trait_name::$method_name(&$v), Method::DefaultRootCtxt);
        },
        (assert_override $v:expr) => {
            assert_eq!($v.method(), Method::OverrideMacroCtxt);
            assert_eq!($trait_name::method(&$v), Method::OverrideMacroCtxt);
            assert_eq!($v.$method_name(), Method::OverrideRootCtxt);
            assert_eq!($trait_name::$method_name(&$v), Method::OverrideRootCtxt);
        },
        (impl for $t:ty) => {
            impl $trait_name for $t {
                fn method(&self) -> Method {
                    Method::OverrideMacroCtxt
                }

                fn $method_name(&self) -> Method {
                    Method::OverrideRootCtxt
                }
            }
        }
    }
}

x!(test_trait, test_trait2, MyTrait, method);

impl MyTrait for char {}
test_trait!(impl for i32);
test_trait2!(impl for i64);

pub fn check_crate_local() {
    test_trait!(check_resolutions);
    test_trait2!(check_resolutions);
}

// Check that any comparison of idents at monomorphization time is correct
pub fn check_crate_local_generic<T: MyTrait, U: MyTrait>(t: T, u: U) {
    test_trait!(check_resolutions);
    test_trait2!(check_resolutions);

    test_trait!(assert_no_override t);
    test_trait2!(assert_no_override t);
    test_trait!(assert_override u);
    test_trait2!(assert_override u);
}
