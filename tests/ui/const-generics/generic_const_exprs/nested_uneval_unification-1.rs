//@ run-pass
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

fn zero_init<const N: usize>() -> Substs1<N>
where
    [u8; N + 1]: ,
{
    Substs1([0; N + 1])
}
struct Substs1<const N: usize>([u8; N + 1])
where
    [(); N + 1]: ;

fn substs2<const M: usize>() -> Substs1<{ M * 2 }>
where
    [(); { M * 2 } + 1]: ,
{
    zero_init::<{ M * 2 }>()
}

fn substs3<const L: usize>() -> Substs1<{ (L - 1) * 2 }>
where
    [(); (L - 1) * 2 + 1]: ,
{
    substs2::<{ L - 1 }>()
}

fn main() {
    assert_eq!(substs3::<2>().0, [0; 3]);
}

// Test that the ``{ (L - 1) * 2 + 1 }`` bound on ``substs3`` satisfies the
// ``{ N + 1 }`` bound on ``Substs1``
