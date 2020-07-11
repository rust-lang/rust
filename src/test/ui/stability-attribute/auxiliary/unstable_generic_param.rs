#![crate_type = "lib"]
#![feature(staged_api)]

#![stable(feature = "stable_test_feature", since = "1.0.0")]

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub trait Trait1<#[unstable(feature = "unstable_default", issue = "none")] T = ()> {
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    fn foo() -> T;
}

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub trait Trait2<#[unstable(feature = "unstable_default", issue = "none")] T = usize> {
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    fn foo() -> T;
}

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub trait Trait3<T = ()> {
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    fn foo() -> T;
}

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub struct Struct1<#[unstable(feature = "unstable_default", issue = "none")] T = usize> {
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    pub field: T,
}

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub struct Struct2<T = usize> {
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    pub field: T,
}

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub struct Struct3<A = isize, #[unstable(feature = "unstable_default", issue = "none")] B = usize> {
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    pub field1: A,
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    pub field2: B,
}

#[rustc_deprecated(since = "1.1.0", reason = "test")]
#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub struct Struct4<A = usize> {
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    pub field: A,
}

#[rustc_deprecated(since = "1.1.0", reason = "test")]
#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub struct Struct5<#[unstable(feature = "unstable_default", issue = "none")] A = usize> {
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    pub field: A,
}

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub struct Struct6<#[unstable(feature = "unstable_default6", issue = "none")] T = usize> {
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    pub field: T,
}

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub const STRUCT1: Struct1 = Struct1 { field: 1 };

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub const STRUCT2: Struct2 = Struct2 { field: 1 };

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub const STRUCT3: Struct3 = Struct3 { field1: 1, field2: 2 };

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub const STRUCT4: Struct4 = Struct4 { field: 1 };

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub const STRUCT5: Struct5 = Struct5 { field: 1 };
