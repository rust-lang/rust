#![feature(rustc_attrs)]

#[rustc_object_lifetime_default]
struct A<
    T, //~ ERROR Empty
>(T);

#[rustc_object_lifetime_default]
struct B<
    'a,
    T, //~ ERROR Empty
>(&'a (), T);

#[rustc_object_lifetime_default]
struct C<
    'a,
    T: 'a, //~ ERROR 'a
>(&'a T);

#[rustc_object_lifetime_default]
struct D<
    'a,
    'b,
    T: 'a + 'b, //~ ERROR Ambiguous
>(&'a T, &'b T);

#[rustc_object_lifetime_default]
struct E<
    'a,
    'b: 'a,
    T: 'b, //~ ERROR 'b
>(&'a T, &'b T);

#[rustc_object_lifetime_default]
struct F<
    'a,
    'b,
    T: 'a, //~ ERROR 'a
    U: 'b, //~ ERROR 'b
>(&'a T, &'b U);

#[rustc_object_lifetime_default]
struct G<
    'a,
    'b,
    T: 'a,      //~ ERROR 'a
    U: 'a + 'b, //~ ERROR Ambiguous
>(&'a T, &'b U);

// Check that we also dump the default for the implicit `Self` type param of traits.
#[rustc_object_lifetime_default]
trait H< //~ ERROR 'a
    'a,
    'b,
    T: 'b, //~ ERROR 'b
>: 'a {}

fn main() {}
