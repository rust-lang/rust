// ignore-compare-mode-nll

#![feature(rustc_attrs)]

#[rustc_object_lifetime_default]
struct A<T>(T); //~ ERROR BaseDefault

#[rustc_object_lifetime_default]
struct B<'a,T>(&'a (), T); //~ ERROR BaseDefault

#[rustc_object_lifetime_default]
struct C<'a,T:'a>(&'a T); //~ ERROR 'a

#[rustc_object_lifetime_default]
struct D<'a,'b,T:'a+'b>(&'a T, &'b T); //~ ERROR Ambiguous

#[rustc_object_lifetime_default]
struct E<'a,'b:'a,T:'b>(&'a T, &'b T); //~ ERROR 'b

#[rustc_object_lifetime_default]
struct F<'a,'b,T:'a,U:'b>(&'a T, &'b U); //~ ERROR 'a,'b

#[rustc_object_lifetime_default]
struct G<'a,'b,T:'a,U:'a+'b>(&'a T, &'b U); //~ ERROR 'a,Ambiguous

fn main() { }
