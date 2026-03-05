#![feature(rustc_attrs)]

#[rustc_dump_object_lifetime_defaults]
struct A<
    T, //~ ERROR BaseDefault
>(T);

#[rustc_dump_object_lifetime_defaults]
struct B<
    'a,
    T, //~ ERROR BaseDefault
>(&'a (), T);

#[rustc_dump_object_lifetime_defaults]
struct C<
    'a,
    T: 'a, //~ ERROR 'a
>(&'a T);

#[rustc_dump_object_lifetime_defaults]
struct D<
    'a,
    'b,
    T: 'a + 'b, //~ ERROR Ambiguous
>(&'a T, &'b T);

#[rustc_dump_object_lifetime_defaults]
struct E<
    'a,
    'b: 'a,
    T: 'b, //~ ERROR 'b
>(&'a T, &'b T);

#[rustc_dump_object_lifetime_defaults]
struct F<
    'a,
    'b,
    T: 'a, //~ ERROR 'a
    U: 'b, //~ ERROR 'b
>(&'a T, &'b U);

#[rustc_dump_object_lifetime_defaults]
struct G<
    'a,
    'b,
    T: 'a,      //~ ERROR 'a
    U: 'a + 'b, //~ ERROR Ambiguous
>(&'a T, &'b U);

fn main() {}
