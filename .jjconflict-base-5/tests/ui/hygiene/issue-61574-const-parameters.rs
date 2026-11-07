// A more comprehensive test that const parameters have correctly implemented
// hygiene

//@ check-pass

use std::ops::Add;

struct VectorLike<T, const SIZE: usize>([T; {SIZE}]);

macro_rules! impl_operator_overload {
    ($trait_ident:ident, $method_ident:ident) => {

        impl<T, const SIZE: usize> $trait_ident for VectorLike<T, {SIZE}>
        where
            T: $trait_ident,
        {
            type Output = VectorLike<T, {SIZE}>;

            fn $method_ident(self, _: VectorLike<T, {SIZE}>) -> VectorLike<T, {SIZE}> {
                let _ = SIZE;
                unimplemented!()
            }
        }

    }
}

impl_operator_overload!(Add, add);

fn main() {}
