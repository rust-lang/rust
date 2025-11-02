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

    #[cfg(all(
        feature = "Win32",
        feature = "Win32_System",
        feature = "Win32_System_Diagnostics",
        feature = "Win32_System_Diagnostics_Debug"
    ))]
    pub fn lot_of_features() {}
}

#[doc(cfg(feature = "foo-method"))]
#[deprecated = "Whatever [`Foo::a_method`](#method.a_method)"]
pub trait Trait {
    type X;
    const Y: u32;

    #[deprecated = "Whatever [`Foo`](#tadam)"]
    fn foo() {}
    fn fooo();
}

impl Trait for Foo {
    type X = u32;
    const Y: u32 = 0;

    fn fooo() {}
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

    /// Short doc.
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

pub struct WhereWhitespace<T>(T);

impl<T> WhereWhitespace<T> {
    pub fn new<F>(f: F) -> Self
    where
        F: FnMut() -> i32,
    {todo!()}
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

pub mod scroll_traits {
    use std::iter::*;

    struct Intersperse<T>(T);
    struct IntersperseWith<T, U>(T, U);
    struct Flatten<T>(T);
    struct Peekable<T>(T);

    /// Shamelessly (partially) copied from `std::iter::Iterator`.
    /// It allows us to check that the scroll is working as expected on "hidden" items.
    pub trait Iterator {
        type Item;

        fn next(&mut self) -> Option<Self::Item>;
        fn size_hint(&self) -> (usize, Option<usize>);
        fn count(self) -> usize
        where
            Self: Sized;
        fn last(self) -> Option<Self::Item>
        where
            Self: Sized;
        fn advance_by(&mut self, n: usize) -> Result<(), usize>;
        fn nth(&mut self, n: usize) -> Option<Self::Item>;
        fn step_by(self, step: usize) -> StepBy<Self>
        where
            Self: Sized;
        fn chain<U>(self, other: U) -> Chain<Self, U::IntoIter>
        where
            Self: Sized,
            U: IntoIterator<Item = Self::Item>;
        fn zip<U>(self, other: U) -> Zip<Self, U::IntoIter>
        where
            Self: Sized,
            U: IntoIterator;
        fn intersperse(self, separator: Self::Item) -> Intersperse<Self>
        where
            Self: Sized,
            Self::Item: Clone;
        fn intersperse_with<G>(self, separator: G) -> IntersperseWith<Self, G>
        where
            Self: Sized,
            G: FnMut() -> Self::Item;
        fn map<B, F>(self, f: F) -> Map<Self, F>
        where
            Self: Sized,
            F: FnMut(Self::Item) -> B;
        fn for_each<F>(self, f: F)
        where
            Self: Sized,
            F: FnMut(Self::Item);
        fn filter<P>(self, predicate: P) -> Filter<Self, P>
        where
            Self: Sized,
            P: FnMut(&Self::Item) -> bool;
        fn filter_map<B, F>(self, f: F) -> FilterMap<Self, F>
        where
            Self: Sized,
            F: FnMut(Self::Item) -> Option<B>;
        fn enumerate(self) -> Enumerate<Self>
        where
            Self: Sized;
        fn peekable(self) -> Peekable<Self>
        where
            Self: Sized;
        fn skip_while<P>(self, predicate: P) -> SkipWhile<Self, P>
        where
            Self: Sized,
            P: FnMut(&Self::Item) -> bool;
        fn take_while<P>(self, predicate: P) -> TakeWhile<Self, P>
        where
            Self: Sized,
            P: FnMut(&Self::Item) -> bool;
        fn map_while<B, P>(self, predicate: P) -> MapWhile<Self, P>
        where
            Self: Sized,
            P: FnMut(Self::Item) -> Option<B>;
        fn skip(self, n: usize) -> Skip<Self>
        where
            Self: Sized;
        fn take(self, n: usize) -> Take<Self>
        where
            Self: Sized;
        fn scan<St, B, F>(self, initial_state: St, f: F) -> Scan<Self, St, F>
        where
            Self: Sized,
            F: FnMut(&mut St, Self::Item) -> Option<B>;
        fn flat_map<U, F>(self, f: F) -> FlatMap<Self, U, F>
        where
            Self: Sized,
            U: IntoIterator,
            F: FnMut(Self::Item) -> U;
        fn flatten(self) -> Flatten<Self>
        where
            Self: Sized,
            Self::Item: IntoIterator;
        fn fuse(self) -> Fuse<Self>
        where
            Self: Sized;
        fn inspect<F>(self, f: F) -> Inspect<Self, F>
        where
            Self: Sized,
            F: FnMut(&Self::Item);
        fn by_ref(&mut self) -> &mut Self
        where
            Self: Sized;
        fn collect<B: FromIterator<Self::Item>>(self) -> B
        where
            Self: Sized;
        fn collect_into<E: Extend<Self::Item>>(self, collection: &mut E) -> &mut E
        where
            Self: Sized;
        fn partition<B, F>(self, f: F) -> (B, B)
        where
            Self: Sized,
            B: Default + Extend<Self::Item>,
            F: FnMut(&Self::Item) -> bool;
        fn partition_in_place<'a, T: 'a, P>(mut self, predicate: P) -> usize
        where
            Self: Sized + DoubleEndedIterator<Item = &'a mut T>,
            P: FnMut(&T) -> bool;
        fn is_partitioned<P>(mut self, mut predicate: P) -> bool
        where
            Self: Sized,
            P: FnMut(Self::Item) -> bool;
        fn fold<B, F>(mut self, init: B, mut f: F) -> B
        where
            Self: Sized,
            F: FnMut(B, Self::Item) -> B;
        fn reduce<F>(mut self, f: F) -> Option<Self::Item>
        where
            Self: Sized,
            F: FnMut(Self::Item, Self::Item) -> Self::Item;
        fn all<F>(&mut self, f: F) -> bool
        where
            Self: Sized,
            F: FnMut(Self::Item) -> bool;
        fn any<F>(&mut self, f: F) -> bool
        where
            Self: Sized,
            F: FnMut(Self::Item) -> bool;
        fn find<P>(&mut self, predicate: P) -> Option<Self::Item>
        where
            Self: Sized,
            P: FnMut(&Self::Item) -> bool;
        fn find_map<B, F>(&mut self, f: F) -> Option<B>
        where
            Self: Sized,
            F: FnMut(Self::Item) -> Option<B>;
        fn position<P>(&mut self, predicate: P) -> Option<usize>
        where
            Self: Sized,
            P: FnMut(Self::Item) -> bool;
        /// We will scroll to "string" to ensure it scrolls as expected.
        fn this_is_a_method_with_a_long_name_returning_something() -> String;
    }

    /// This one doesn't have hidden items (because there are too many) so we can also confirm that it
    /// scrolls as expected.
    pub trait TraitWithLongItemsName {
        fn this_is_a_method_with_a_long_name_returning_something() -> String;
    }
}

pub struct Derefer(String);

impl std::ops::Deref for Derefer {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
