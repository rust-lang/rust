// Regression test for the issue #50716: NLL ignores lifetimes bounds
// derived from `Sized` requirements

trait A {
    type X: ?Sized;
}

fn foo<'a, T: 'static>(s: Box<<&'a T as A>::X>)
where
    for<'b> &'b T: A,
    <&'static T as A>::X: Sized
{
    let _x = *s; //~ ERROR
}

fn main() {}
