#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

trait One: Sized {
    const ONE: Self;
}

const fn noop<T: One>(a: &mut T, b: &mut T) {
    _ = (a, b);
}

struct Test<T: One, const LEN: usize>([T; LEN * LEN])
where
    [u8; LEN * LEN]:;

impl<T: One, const LEN: usize> Test<T, LEN>
where
    [u8; LEN * LEN]:,
{
    const fn test() -> Self {
        let mut a = Self([T::ONE; LEN * LEN]);
        let mut i = 0;
        while i < LEN {
            let mut one = T::ONE; //~ ERROR destructor of `T` cannot be evaluated at compile-time
            noop(&mut one, &mut a.0[i * i + 1]);
            i += 1;
        }
        a
    }
}

fn main() {}