//@aux-build:proc_macros.rs
//@no-rustfix
#![warn(clippy::unnecessary_fallible_conversions)]

extern crate proc_macros;

struct Foo;
impl TryFrom<i32> for Foo {
    type Error = ();
    fn try_from(_: i32) -> Result<Self, Self::Error> {
        Ok(Foo)
    }
}
impl From<i64> for Foo {
    fn from(_: i64) -> Self {
        Foo
    }
}

fn main() {
    // `Foo` only implements `TryFrom<i32>` and not `From<i32>`, so don't lint
    let _: Result<Foo, _> = 0i32.try_into();
    let _: Result<Foo, _> = i32::try_into(0i32);
    let _: Result<Foo, _> = Foo::try_from(0i32);

    // ... it does impl From<i64> however
    let _: Result<Foo, _> = 0i64.try_into();
    //~^ ERROR: use of a fallible conversion when an infallible one could be used
    let _: Result<Foo, _> = i64::try_into(0i64);
    //~^ ERROR: use of a fallible conversion when an infallible one could be used
    let _: Result<Foo, _> = Foo::try_from(0i64);
    //~^ ERROR: use of a fallible conversion when an infallible one could be used

    let _: Result<i64, _> = 0i32.try_into();
    //~^ ERROR: use of a fallible conversion when an infallible one could be used
    let _: Result<i64, _> = i32::try_into(0i32);
    //~^ ERROR: use of a fallible conversion when an infallible one could be used
    let _: Result<i64, _> = <_>::try_from(0i32);
    //~^ ERROR: use of a fallible conversion when an infallible one could be used

    // From a macro
    let _: Result<i64, _> = proc_macros::external!(0i32).try_into();
}
