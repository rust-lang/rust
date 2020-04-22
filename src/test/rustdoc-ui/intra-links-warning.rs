// check-pass

       //! Test with [Foo::baz], [Bar::foo], ...
//~^ WARNING `[Foo::baz]` cannot be resolved
//~| WARNING `[Bar::foo]` cannot be resolved
     //! , [Uniooon::X] and [Qux::Z].
//~^ WARNING `[Uniooon::X]` cannot be resolved
//~| WARNING `[Qux::Z]` cannot be resolved
       //!
      //! , [Uniooon::X] and [Qux::Z].
//~^ WARNING `[Uniooon::X]` cannot be resolved
//~| WARNING `[Qux::Z]` cannot be resolved

       /// [Qux:Y]
//~^ WARNING `[Qux:Y]` cannot be resolved
pub struct Foo {
    pub bar: usize,
}

/// Foo
/// bar [BarA] bar //~ WARNING `[BarA]` cannot be resolved
/// baz
pub fn a() {}

/**
 * Foo
 * bar [BarB] bar //~ WARNING `[BarB]` cannot be resolved
 * baz
 */
pub fn b() {}

/** Foo

bar [BarC] bar //~ WARNING `[BarC]` cannot be resolved
baz

    let bar_c_1 = 0;
    let bar_c_2 = 0;
    let g = [bar_c_1];
    let h = g[bar_c_2];

*/
pub fn c() {}

#[doc = "Foo\nbar [BarD] bar\nbaz"] //~ WARNING `[BarD]` cannot be resolved
pub fn d() {}

macro_rules! f {
    ($f:expr) => {
        #[doc = $f] //~ WARNING `[BarF]` cannot be resolved
        pub fn f() {}
    }
}
f!("Foo\nbar [BarF] bar\nbaz");

/** # for example,
 *
 * time to introduce a link [error]*/ //~ WARNING `[error]` cannot be resolved
pub struct A;

/**
 * # for example,
 *
 * time to introduce a link [error] //~ WARNING `[error]` cannot be resolved
 */
pub struct B;

#[doc = "single line [error]"] //~ WARNING `[error]` cannot be resolved
pub struct C;

#[doc = "single line with \"escaping\" [error]"] //~ WARNING `[error]` cannot be resolved
pub struct D;

/// Item docs. //~ WARNING `[error]` cannot be resolved
#[doc="Hello there!"]
/// [error]
pub struct E;

///
/// docs [error1] //~ WARNING `[error1]` cannot be resolved

/// docs [error2] //~ WARNING `[error2]` cannot be resolved
///
pub struct F;
