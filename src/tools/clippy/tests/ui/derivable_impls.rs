#![allow(dead_code)]

use std::collections::HashMap;

struct FooDefault<'a> {
    a: bool,
    b: i32,
    c: u64,
    d: Vec<i32>,
    e: FooND1,
    f: FooND2,
    g: HashMap<i32, i32>,
    h: (i32, Vec<i32>),
    i: [Vec<i32>; 3],
    j: [i32; 5],
    k: Option<i32>,
    l: &'a [i32],
}

impl std::default::Default for FooDefault<'_> {
    fn default() -> Self {
        Self {
            a: false,
            b: 0,
            c: 0u64,
            d: vec![],
            e: Default::default(),
            f: FooND2::default(),
            g: HashMap::new(),
            h: (0, vec![]),
            i: [vec![], vec![], vec![]],
            j: [0; 5],
            k: None,
            l: &[],
        }
    }
}

struct TupleDefault(bool, i32, u64);

impl std::default::Default for TupleDefault {
    fn default() -> Self {
        Self(false, 0, 0u64)
    }
}

struct FooND1 {
    a: bool,
}

impl std::default::Default for FooND1 {
    fn default() -> Self {
        Self { a: true }
    }
}

struct FooND2 {
    a: i32,
}

impl std::default::Default for FooND2 {
    fn default() -> Self {
        Self { a: 5 }
    }
}

struct FooNDNew {
    a: bool,
}

impl FooNDNew {
    fn new() -> Self {
        Self { a: true }
    }
}

impl Default for FooNDNew {
    fn default() -> Self {
        Self::new()
    }
}

struct FooNDVec(Vec<i32>);

impl Default for FooNDVec {
    fn default() -> Self {
        Self(vec![5, 12])
    }
}

struct StrDefault<'a>(&'a str);

impl Default for StrDefault<'_> {
    fn default() -> Self {
        Self("")
    }
}

#[derive(Default)]
struct AlreadyDerived(i32, bool);

macro_rules! mac {
    () => {
        0
    };
    ($e:expr) => {
        struct X(u32);
        impl Default for X {
            fn default() -> Self {
                Self($e)
            }
        }
    };
}

mac!(0);

struct Y(u32);
impl Default for Y {
    fn default() -> Self {
        Self(mac!())
    }
}

struct RustIssue26925<T> {
    a: Option<T>,
}

// We should watch out for cases where a manual impl is needed because a
// derive adds different type bounds (https://github.com/rust-lang/rust/issues/26925).
// For example, a struct with Option<T> does not require T: Default, but a derive adds
// that type bound anyways. So until #26925 get fixed we should disable lint
// for the following case
impl<T> Default for RustIssue26925<T> {
    fn default() -> Self {
        Self { a: None }
    }
}

struct SpecializedImpl<A, B> {
    a: A,
    b: B,
}

impl<T: Default> Default for SpecializedImpl<T, T> {
    fn default() -> Self {
        Self {
            a: T::default(),
            b: T::default(),
        }
    }
}

struct WithoutSelfCurly {
    a: bool,
}

impl Default for WithoutSelfCurly {
    fn default() -> Self {
        WithoutSelfCurly { a: false }
    }
}

struct WithoutSelfParan(bool);

impl Default for WithoutSelfParan {
    fn default() -> Self {
        WithoutSelfParan(false)
    }
}

// https://github.com/rust-lang/rust-clippy/issues/7655

pub struct SpecializedImpl2<T> {
    v: Vec<T>,
}

impl Default for SpecializedImpl2<String> {
    fn default() -> Self {
        Self { v: Vec::new() }
    }
}

// https://github.com/rust-lang/rust-clippy/issues/7654

pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

/// `#000000`
impl Default for Color {
    fn default() -> Self {
        Color { r: 0, g: 0, b: 0 }
    }
}

pub struct Color2 {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Default for Color2 {
    /// `#000000`
    fn default() -> Self {
        Self { r: 0, g: 0, b: 0 }
    }
}

pub struct RepeatDefault1 {
    a: [i8; 32],
}

impl Default for RepeatDefault1 {
    fn default() -> Self {
        RepeatDefault1 { a: [0; 32] }
    }
}

pub struct RepeatDefault2 {
    a: [i8; 33],
}

impl Default for RepeatDefault2 {
    fn default() -> Self {
        RepeatDefault2 { a: [0; 33] }
    }
}

// https://github.com/rust-lang/rust-clippy/issues/7753

pub enum IntOrString {
    Int(i32),
    String(String),
}

impl Default for IntOrString {
    fn default() -> Self {
        IntOrString::Int(0)
    }
}

pub enum SimpleEnum {
    Foo,
    Bar,
}

impl Default for SimpleEnum {
    fn default() -> Self {
        SimpleEnum::Bar
    }
}

pub enum NonExhaustiveEnum {
    Foo,
    #[non_exhaustive]
    Bar,
}

impl Default for NonExhaustiveEnum {
    fn default() -> Self {
        NonExhaustiveEnum::Bar
    }
}

// https://github.com/rust-lang/rust-clippy/issues/10396

#[derive(Default)]
struct DefaultType;

struct GenericType<T = DefaultType> {
    t: T,
}

impl Default for GenericType {
    fn default() -> Self {
        Self { t: Default::default() }
    }
}

struct InnerGenericType<T> {
    t: T,
}

impl Default for InnerGenericType<DefaultType> {
    fn default() -> Self {
        Self { t: Default::default() }
    }
}

struct OtherGenericType<T = DefaultType> {
    inner: InnerGenericType<T>,
}

impl Default for OtherGenericType {
    fn default() -> Self {
        Self {
            inner: Default::default(),
        }
    }
}

mod issue10158 {
    pub trait T {}

    #[derive(Default)]
    pub struct S {}
    impl T for S {}

    pub struct Outer {
        pub inner: Box<dyn T>,
    }

    impl Default for Outer {
        fn default() -> Self {
            Outer {
                // Box::<S>::default() adjusts to Box<dyn T>
                inner: Box::<S>::default(),
            }
        }
    }
}

mod issue11368 {
    pub struct A {
        a: u32,
    }

    impl Default for A {
        #[track_caller]
        fn default() -> Self {
            Self { a: 0 }
        }
    }
}

fn main() {}
