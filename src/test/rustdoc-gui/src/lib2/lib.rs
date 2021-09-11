// ignore-tidy-linelength

#![feature(doc_cfg)]

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
    pub fn a_method(&self) {}
}

pub trait Trait {
    type X;
    const Y: u32;

    fn foo() {}
}

impl Trait for Foo {
    type X = u32;
    const Y: u32 = 0;
}


impl implementors::Whatever for Foo {
    type Foo = u32;
}

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

    pub trait ALongNameBecauseItHelpsTestingTheCurrentProblem: DerefMut<Target = u32>
        + From<u128> + Send + Sync + AsRef<str> + 'static {}
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
