// check-pass

       //! Test with [Foo::baz], [Bar::foo], ...
//~^ WARNING `Foo::baz`
//~| WARNING `Bar::foo`
     //! , [Uniooon::X] and [Qux::Z].
//~^ WARNING `Uniooon::X`
//~| WARNING `Qux::Z`
       //!
      //! , [Uniooon::X] and [Qux::Z].
//~^ WARNING `Uniooon::X`
//~| WARNING `Qux::Z`

       /// [Qux:Y]
//~^ WARNING `Qux:Y`
pub struct Foo {
    pub bar: usize,
}

/// Foo
/// bar [BarA] bar //~ WARNING `BarA`
/// baz
pub fn a() {}

/**
 * Foo
 * bar [BarB] bar //~ WARNING `BarB`
 * baz
 */
pub fn b() {}

/** Foo

bar [BarC] bar //~ WARNING `BarC`
baz

    let bar_c_1 = 0;
    let bar_c_2 = 0;
    let g = [bar_c_1];
    let h = g[bar_c_2];

*/
pub fn c() {}

#[doc = "Foo\nbar [BarD] bar\nbaz"] //~ WARNING `BarD`
pub fn d() {}

macro_rules! f {
    ($f:expr) => {
        #[doc = $f] //~ WARNING `BarF`
        pub fn f() {}
    }
}
f!("Foo\nbar [BarF] bar\nbaz");

/** # for example,
 *
 * time to introduce a link [error]*/ //~ WARNING `error`
pub struct A;

/**
 * # for example,
 *
 * time to introduce a link [error] //~ WARNING `error`
 */
pub struct B;

#[doc = "single line [error]"] //~ WARNING `error`
pub struct C;

#[doc = "single line with \"escaping\" [error]"] //~ WARNING `error`
pub struct D;

/// Item docs. //~ WARNING `error`
#[doc="Hello there!"]
/// [error]
pub struct E;

///
/// docs [error1] //~ WARNING `error1`

/// docs [error2] //~ WARNING `error2`
///
pub struct F;
