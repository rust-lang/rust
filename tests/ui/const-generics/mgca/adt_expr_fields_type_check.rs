#![feature(min_generic_const_args, adt_const_params)]
#![expect(incomplete_features)]

#[derive(Eq, PartialEq, std::marker::ConstParamTy)]
struct S1<T> {
    f1: T,
    f2: isize,
}

#[derive(Eq, PartialEq, std::marker::ConstParamTy)]
struct S2<T>(T, isize);

#[derive(Eq, PartialEq, std::marker::ConstParamTy)]
enum En<T> {
    Var1(bool, T),
    Var2 { field: i64 },
}

fn accepts_1<const N: S1<u8>>() {}
fn accepts_2<const N: S2<u8>>() {}
fn accepts_3<const N: En<u8>>() {}

fn bar<const N: bool>() {
    accepts_1::<{ S1::<u8> { f1: N, f2: N } }>();
    //~^ ERROR the constant `N` is not of type `u8`
    //~| ERROR the constant `N` is not of type `isize`
    accepts_2::<{ S2::<u8>(N, N) }>();
    //~^ ERROR the constant `N` is not of type `u8`
    //~| ERROR the constant `N` is not of type `isize`
    accepts_3::<{ En::Var1::<u8>(N, N) }>();
    //~^ ERROR the constant `N` is not of type `u8`
    accepts_3::<{ En::Var2::<u8> { field: N } }>();
    //~^ ERROR the constant `N` is not of type `i64`
    accepts_3::<{ En::Var2::<u8> { field: const { false } } }>();
    //~^ ERROR mismatched types
}

fn main() {}
