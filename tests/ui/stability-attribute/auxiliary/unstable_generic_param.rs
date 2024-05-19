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

#[deprecated(since = "1.1.0", note = "test")]
#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub struct Struct4<A = usize> {
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    pub field: A,
}

#[deprecated(since = "1.1.0", note = "test")]
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

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub enum Enum1<#[unstable(feature = "unstable_default", issue = "none")] T = usize> {
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    Some(#[stable(feature = "stable_test_feature", since = "1.0.0")] T),
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    None,
}

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub enum Enum2<T = usize> {
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    Some(#[stable(feature = "stable_test_feature", since = "1.0.0")] T),
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    None,
}

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub enum Enum3<T = isize, #[unstable(feature = "unstable_default", issue = "none")] E = usize> {
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    Ok(#[stable(feature = "stable_test_feature", since = "1.0.0")] T),
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    Err(#[stable(feature = "stable_test_feature", since = "1.0.0")] E),
}

#[deprecated(since = "1.1.0", note = "test")]
#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub enum Enum4<T = usize> {
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    Some(#[stable(feature = "stable_test_feature", since = "1.0.0")] T),
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    None,
}

#[deprecated(since = "1.1.0", note = "test")]
#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub enum Enum5<#[unstable(feature = "unstable_default", issue = "none")] T = usize> {
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    Some(#[stable(feature = "stable_test_feature", since = "1.0.0")] T),
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    None,
}

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub enum Enum6<#[unstable(feature = "unstable_default6", issue = "none")] T = usize> {
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    Some(#[stable(feature = "stable_test_feature", since = "1.0.0")] T),
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    None,
}

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub const ENUM1: Enum1 = Enum1::Some(1);

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub const ENUM2: Enum2 = Enum2::Some(1);

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub const ENUM3: Enum3 = Enum3::Ok(1);
#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub const ENUM3B: Enum3 = Enum3::Err(1);

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub const ENUM4: Enum4 = Enum4::Some(1);

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub const ENUM5: Enum5 = Enum5::Some(1);

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub type Alias1<#[unstable(feature = "unstable_default", issue = "none")] T = usize> = Option<T>;

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub type Alias2<T = usize> = Option<T>;

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub type Alias3<T = isize, #[unstable(feature = "unstable_default", issue = "none")] E = usize> =
    Result<T, E>;

#[deprecated(since = "1.1.0", note = "test")]
#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub type Alias4<T = usize> = Option<T>;

#[deprecated(since = "1.1.0", note = "test")]
#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub type Alias5<#[unstable(feature = "unstable_default", issue = "none")] T = usize> = Option<T>;

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub type Alias6<#[unstable(feature = "unstable_default6", issue = "none")] T = usize> = Option<T>;

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub const ALIAS1: Alias1 = Alias1::Some(1);

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub const ALIAS2: Alias2 = Alias2::Some(1);

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub const ALIAS3: Alias3 = Alias3::Ok(1);
#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub const ALIAS3B: Alias3 = Alias3::Err(1);

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub const ALIAS4: Alias4 = Alias4::Some(1);

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub const ALIAS5: Alias5 = Alias5::Some(1);


#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub trait Alloc {}

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub struct System {}

#[stable(feature = "stable_test_feature", since = "1.0.0")]
impl Alloc for System {}

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub struct Box1<T, #[unstable(feature = "box_alloc_param", issue = "none")] A: Alloc = System> {
    ptr: *mut T,
    alloc: A,
}

impl<T> Box1<T, System> {
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    pub fn new(mut t: T) -> Self {
        unsafe { Self { ptr: &mut t, alloc: System {} } }
    }
}

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub struct Box2<T, A: Alloc = System> {
    ptr: *mut T,
    alloc: A,
}

impl<T> Box2<T, System> {
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    pub fn new(mut t: T) -> Self {
        Self { ptr: &mut t, alloc: System {} }
    }
}

#[stable(feature = "stable_test_feature", since = "1.0.0")]
pub struct Box3<T> {
    ptr: *mut T,
}

impl<T> Box3<T> {
    #[stable(feature = "stable_test_feature", since = "1.0.0")]
    pub fn new(mut t: T) -> Self {
        Self { ptr: &mut t }
    }
}
