// This is a regression test for one of the problems in #128887; it checks that the
// strategy in #129714 avoids trait solver overflows in this specific case.

// skip-filecheck
//@ compile-flags: -Zinline-mir

pub trait Foo {
    type Associated;
    type Chain: Foo<Associated = Self::Associated>;
}

trait FooExt {
    fn do_ext() {}
}
impl<T: Foo<Associated = f64>> FooExt for T {}

#[allow(unconditional_recursion)]
fn recurse<T: Foo<Associated = f64>>() {
    T::do_ext();
    recurse::<T::Chain>();
}

macro_rules! emit {
    ($($m:ident)*) => {$(
        pub fn $m<T: Foo<Associated = f64>>() {
            recurse::<T>();
        }
    )*}
}

// Increase the chance of triggering the bug
emit!(m00 m01 m02 m03 m04 m05 m06 m07 m08 m09 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19);
