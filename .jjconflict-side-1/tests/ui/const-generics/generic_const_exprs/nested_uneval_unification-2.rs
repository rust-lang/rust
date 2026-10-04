//@ run-pass
#![feature(generic_const_exprs)]
#![allow(incomplete_features, unused_parens, unused_braces)]

fn zero_init<const N: usize>() -> Substs1<{{ N }}>
where
    [u8; {{ N }}]: ,
{
    Substs1([0; {{ N }}])
}

struct Substs1<const N: usize>([u8; {{ N }}])
where
    [(); {{ N }}]: ;

fn substs2<const M: usize>() -> Substs1<{{ M }}> {
    zero_init::<{{ M }}>()
}

fn substs3<const L: usize>() -> Substs1<{{ L }}> {
    substs2::<{{ L }}>()
}

fn main() {
    assert_eq!(substs3::<2>().0, [0; 2]);
}

// Test that the implicit ``{{ L }}`` bound on ``substs3`` satisfies the
// ``{{ N }}`` bound on ``Substs1``
// FIXME(generic_const_exprs): come up with a less brittle test for this using assoc consts
// once normalization is implemented for them.
