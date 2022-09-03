// ignore-tidy-linelength

#![feature(doc_cfg)]

pub mod another_folder;
pub mod another_mod;

pub mod module {
    pub mod sub_module {
        pub mod sub_sub_module {
            pub fn foo() {}
        }
        pub fn bar() {}
    }
    pub fn whatever() {}
}

pub fn foobar() {}

pub type Alias = u32;

#[doc(cfg(feature = "foo-method"))]
pub struct Foo {
    pub x: Alias,
}

impl Foo {
    /// Some documentation
    /// # A Heading
    pub fn a_method(&self) {}
}

#[doc(cfg(feature = "foo-method"))]
#[deprecated = "Whatever [`Foo::a_method`](#method.a_method)"]
pub trait Trait {
    type X;
    const Y: u32;

    #[deprecated = "Whatever [`Foo`](#tadam)"]
    fn foo() {}
}

impl Trait for Foo {
    type X = u32;
    const Y: u32 = 0;
}

impl implementors::Whatever for Foo {
    type Foo = u32;
}

#[doc(inline)]
pub use implementors::TraitToReexport;

pub struct StructToImplOnReexport;

impl TraitToReexport for StructToImplOnReexport {}

pub mod sub_mod {
    /// ```txt
    /// aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    /// ```
    ///
    /// ```
    /// aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    /// ```
    pub struct Foo;
}

pub mod long_trait {
    use std::ops::DerefMut;

    pub trait ALongNameBecauseItHelpsTestingTheCurrentProblem:
        DerefMut<Target = u32> + From<u128> + Send + Sync + AsRef<str> + 'static
    {
    }
}

pub mod long_table {
    /// | This::is::a::kinda::very::long::header::number::one | This::is::a::kinda::very::long::header::number::two | This::is::a::kinda::very::long::header::number::one | This::is::a::kinda::very::long::header::number::two |
    /// | ----------- | ----------- | ----------- | ----------- |
    /// | This::is::a::kinda::long::content::number::one | This::is::a::kinda::very::long::content::number::two | This::is::a::kinda::long::content::number::one | This::is::a::kinda::very::long::content::number::two |
    ///
    /// I wanna sqdkfnqds f dsqf qds f dsqf dsq f dsq f qds f qds f qds f dsqq f dsf sqdf dsq fds f dsq f dq f ds fq sd fqds f dsq f sqd fsq df sd fdsqfqsd fdsq f dsq f dsqfd s dfq
    pub struct Foo;

    /// | This::is::a::kinda::very::long::header::number::one | This::is::a::kinda::very::long::header::number::two | This::is::a::kinda::very::long::header::number::one | This::is::a::kinda::very::long::header::number::two |
    /// | ----------- | ----------- | ----------- | ----------- |
    /// | This::is::a::kinda::long::content::number::one | This::is::a::kinda::very::long::content::number::two | This::is::a::kinda::long::content::number::one | This::is::a::kinda::very::long::content::number::two |
    ///
    /// I wanna sqdkfnqds f dsqf qds f dsqf dsq f dsq f qds f qds f qds f dsqq f dsf sqdf dsq fds f dsq f dq f ds fq sd fqds f dsq f sqd fsq df sd fdsqfqsd fdsq f dsq f dsqfd s dfq
    impl Foo {
        pub fn foo(&self) {}
    }
}

pub mod summary_table {
    /// | header 1 | header 2 |
    /// | -------- | -------- |
    /// | content | content |
    pub struct Foo;
}

pub mod too_long {
    pub type ReallyLongTypeNameLongLongLong =
        Option<unsafe extern "C" fn(a: *const u8, b: *const u8) -> *const u8>;

    pub const ReallyLongTypeNameLongLongLongConstBecauseWhyNotAConstRightGigaGigaSupraLong: u32 = 0;

    /// This also has a really long doccomment. Lorem ipsum dolor sit amet,
    /// consectetur adipiscing elit. Suspendisse id nibh malesuada, hendrerit
    /// massa vel, tincidunt est. Nulla interdum, sem ac efficitur ornare, arcu
    /// nunc dignissim nibh, at rutrum diam augue ac mauris. Fusce tincidunt et
    /// ligula sed viverra. Aenean sed facilisis dui, non volutpat felis. In
    /// vitae est dui. Donec felis nibh, blandit at nibh eu, tempor suscipit
    /// nisl. Vestibulum ornare porta libero, eu faucibus purus iaculis ut. Ut
    /// quis tincidunt nunc, in mollis purus. Nulla sed interdum quam. Nunc
    /// vitae cursus ex.
    pub struct SuperIncrediblyLongLongLongLongLongLongLongGigaGigaGigaMegaLongLongLongStructName {
        pub a: u32,
    }

    impl SuperIncrediblyLongLongLongLongLongLongLongGigaGigaGigaMegaLongLongLongStructName {
        /// ```
        /// let x = SuperIncrediblyLongLongLongLongLongLongLongGigaGigaGigaMegaLongLongLongStructName { a: 0 };
        /// ```
        pub fn foo(&self) {}
    }
}

pub struct HasALongTraitWithParams {}

pub trait LongTraitWithParamsBananaBananaBanana<T> {}

impl LongTraitWithParamsBananaBananaBanana<usize> for HasALongTraitWithParams {}

#[doc(cfg(any(target_os = "android", target_os = "linux", target_os = "emscripten", target_os = "dragonfly", target_os = "freebsd", target_os = "netbsd", target_os = "openbsd")))]
pub struct LongItemInfo;

pub trait SimpleTrait {}
pub struct LongItemInfo2;

/// Some docs.
#[doc(cfg(any(target_os = "android", target_os = "linux", target_os = "emscripten", target_os = "dragonfly", target_os = "freebsd", target_os = "netbsd", target_os = "openbsd")))]
impl SimpleTrait for LongItemInfo2 {}

pub struct WhereWhitespace<T>;

impl<T> WhereWhitespace<T> {
    pub fn new<F>(f: F) -> Self
    where
        F: FnMut() -> i32,
    {}
}

impl<K, T> Whitespace<&K> for WhereWhitespace<T>
where
    K: std::fmt::Debug,
{
    type Output = WhereWhitespace<T>;
    fn index(&self, _key: &K) -> &Self::Output {
        self
    }
}

pub trait Whitespace<Idx>
where
    Idx: ?Sized,
{
    type Output;
    fn index(&self, index: Idx) -> &Self::Output;
}

pub struct ItemInfoAlignmentTest;

impl ItemInfoAlignmentTest {
    /// This method has docs
    #[deprecated]
    pub fn foo() {}
    #[deprecated]
    pub fn bar() {}
}
