//@ check-pass
//@ revisions: full
// This is a regression test for an ICE in const qualifs where
// `Const::Ty` containing `ty::ConstKind::Unevaluated` was not handled.
// The pattern arises with `generic_const_exprs` and const fn using
// array lengths like `LEN * LEN` and repeat expressions.

#![cfg_attr(full, feature(generic_const_exprs))]
#![cfg_attr(full, allow(incomplete_features))]

trait One: Sized + Copy {
    const ONE: Self;
}

const fn noop<T: One>(a: &mut T, b: &mut T) {
    let _ = (a, b);
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
            let mut one = T::ONE;
            noop(&mut one, &mut a.0[i * i + 1]);
            i += 1;
        }
        a
    }
}

fn main() {}
