#![feature(associated_const_equality, generic_const_items)]
#![expect(incomplete_features)]

// When we type check `main` we wind up having to prove `<() as Trait>::C<128_u64> = 128_u64`
// doing this requires normalizing `<() as Trait>::C<128_u64>`. Previously we did not check
// that consts are well formed before evaluating them (rust-lang/project-const-generics#37) so
// in attempting to get the normalized form of `<() as Trait>::C<128_u64>` we would invoke the
// ctfe machinery on a not-wf type system constant.

trait Trait {
    const C<const N: u32>: u32;
}

impl Trait for () {
    const C<const N: u32>: u32 = N;
}

fn ice<const N: u64, T: Trait<C<N> = { N }>>(_: T) {}
//~^ ERROR: the constant `N` is not of type `u32`

fn main() {
    ice::<128, _>(());
    //~^ ERROR: type mismatch resolving `<() as Trait>::C<128> == 128`
}
