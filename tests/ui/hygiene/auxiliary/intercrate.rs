#![feature(decl_macro)]

pub mod foo {
    pub use self::bar::m;
    mod bar {
        fn f() -> u32 { 1 }
        pub macro m() {
            f()
        }
    }
}

pub struct SomeType;

// `$crate`
pub macro uses_dollar_crate_modern() {
    type Alias = $crate::SomeType;
}

pub macro define_uses_dollar_crate_modern_nested($uses_dollar_crate_modern_nested: ident) {
    macro $uses_dollar_crate_modern_nested() {
        type AliasCrateModernNested = $crate::SomeType;
    }
}

#[macro_export]
macro_rules! define_uses_dollar_crate_legacy_nested {
    () => {
        macro_rules! uses_dollar_crate_legacy_nested {
            () => {
                type AliasLegacyNested = $crate::SomeType;
            }
        }
    }
}

// `crate`
pub macro uses_crate_modern() {
    type AliasCrate = crate::SomeType;
}

pub macro define_uses_crate_modern_nested($uses_crate_modern_nested: ident) {
    macro $uses_crate_modern_nested() {
        type AliasCrateModernNested = crate::SomeType;
    }
}
