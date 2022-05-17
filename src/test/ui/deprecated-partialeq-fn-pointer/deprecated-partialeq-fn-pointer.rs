// check-pass
// revisions: normal chalk
// [chalk]compile-flags: -Zchalk

#![warn(deprecated_in_future)]

fn test() {}

trait Bla {
    fn foo();
}
impl Bla for u32 {
    fn foo() {
        let x: fn() = main;
        let y: fn() = test;
        let _ = x == y; //~ WARN FIXME(skippy) PartialEq on function pointers has been deprecated.
    }
}

trait Bla2<T>
where
    T: PartialEq,
{
}
impl Bla2<fn()> for u32 {} //~ WARN FIXME(skippy) PartialEq on function pointers has been deprecated.

fn main() {
    let x: fn() = main;
    let y: fn() = test;
    let _ = x == y; //~ WARN FIXME(skippy) PartialEq on function pointers has been deprecated.
}
