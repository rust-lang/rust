//@ check-pass

#![feature(adt_const_params, generic_const_parameter_types)]
#![expect(incomplete_features)]

struct Bar<const N: usize, const M: [u8; N]>;

fn foo<const N: usize, const M: [u8; N]>(_: Bar<N, M>) {}

fn main() {
    foo(Bar::<2, { [1; 2] }>);
    foo::<_, _>(Bar::<2, { [1; 2] }>);
}
