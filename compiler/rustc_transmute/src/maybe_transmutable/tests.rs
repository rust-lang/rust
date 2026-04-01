extern crate test;

use itertools::Itertools;

use super::query_context::test::{Def, UltraMinimal};
use crate::{Answer, Assume, Condition, Reason, layout};

type Tree = layout::Tree<Def, !, !>;
type Dfa = layout::Dfa<!, !>;

trait Representation {
    fn is_transmutable(src: Self, dst: Self, assume: Assume) -> Answer<!, !>;
}

impl Representation for Tree {
    fn is_transmutable(src: Self, dst: Self, assume: Assume) -> Answer<!, !> {
        crate::maybe_transmutable::MaybeTransmutableQuery::new(
            src,
            dst,
            assume,
            UltraMinimal::default(),
        )
        .answer()
    }
}

impl Representation for Dfa {
    fn is_transmutable(src: Self, dst: Self, assume: Assume) -> Answer<!, !> {
        crate::maybe_transmutable::MaybeTransmutableQuery::new(
            src,
            dst,
            assume,
            UltraMinimal::default(),
        )
        .answer()
    }
}

fn is_transmutable<R: Representation + Clone>(
    src: &R,
    dst: &R,
    assume: Assume,
) -> crate::Answer<!, !> {
    let src = src.clone();
    let dst = dst.clone();
    // The only dimension of the transmutability analysis we want to test
    // here is the safety analysis. To ensure this, we disable all other
    // toggleable aspects of the transmutability analysis.
    R::is_transmutable(src, dst, assume)
}

mod safety {
    use super::*;
    use crate::Answer;

    const DST_HAS_SAFETY_INVARIANTS: Answer<!, !> =
        Answer::No(crate::Reason::DstMayHaveSafetyInvariants);

    #[test]
    fn src_safe_dst_safe() {
        let src = Tree::Def(Def::NoSafetyInvariants).then(Tree::u8());
        let dst = Tree::Def(Def::NoSafetyInvariants).then(Tree::u8());
        assert_eq!(is_transmutable(&src, &dst, Assume::default()), Answer::Yes);
        assert_eq!(
            is_transmutable(&src, &dst, Assume { safety: true, ..Assume::default() }),
            Answer::Yes
        );
    }

    #[test]
    fn src_safe_dst_unsafe() {
        let src = Tree::Def(Def::NoSafetyInvariants).then(Tree::u8());
        let dst = Tree::Def(Def::HasSafetyInvariants).then(Tree::u8());
        assert_eq!(is_transmutable(&src, &dst, Assume::default()), DST_HAS_SAFETY_INVARIANTS);
        assert_eq!(
            is_transmutable(&src, &dst, Assume { safety: true, ..Assume::default() }),
            Answer::Yes
        );
    }

    #[test]
    fn src_unsafe_dst_safe() {
        let src = Tree::Def(Def::HasSafetyInvariants).then(Tree::u8());
        let dst = Tree::Def(Def::NoSafetyInvariants).then(Tree::u8());
        assert_eq!(is_transmutable(&src, &dst, Assume::default()), Answer::Yes);
        assert_eq!(
            is_transmutable(&src, &dst, Assume { safety: true, ..Assume::default() }),
            Answer::Yes
        );
    }

    #[test]
    fn src_unsafe_dst_unsafe() {
        let src = Tree::Def(Def::HasSafetyInvariants).then(Tree::u8());
        let dst = Tree::Def(Def::HasSafetyInvariants).then(Tree::u8());
        assert_eq!(is_transmutable(&src, &dst, Assume::default()), DST_HAS_SAFETY_INVARIANTS);
        assert_eq!(
            is_transmutable(&src, &dst, Assume { safety: true, ..Assume::default() }),
            Answer::Yes
        );
    }
}

mod size {
    use super::*;

    #[test]
    fn size() {
        let small = Tree::number(1);
        let large = Tree::number(2);

        for alignment in [false, true] {
            for lifetimes in [false, true] {
                for safety in [false, true] {
                    for validity in [false, true] {
                        let assume = Assume { alignment, lifetimes, safety, validity };
                        assert_eq!(
                            is_transmutable(&small, &large, assume),
                            Answer::No(Reason::DstIsTooBig),
                            "assume: {assume:?}"
                        );
                        assert_eq!(
                            is_transmutable(&large, &small, assume),
                            Answer::Yes,
                            "assume: {assume:?}"
                        );
                    }
                }
            }
        }
    }
}

mod bool {
    use super::*;

    #[test]
    fn should_permit_identity_transmutation_tree() {
        let src = Tree::bool();
        assert_eq!(is_transmutable(&src, &src, Assume::default()), Answer::Yes);
        assert_eq!(
            is_transmutable(&src, &src, Assume { validity: true, ..Assume::default() }),
            Answer::Yes
        );
    }

    #[test]
    fn should_permit_identity_transmutation_dfa() {
        let src = Dfa::bool();
        assert_eq!(is_transmutable(&src, &src, Assume::default()), Answer::Yes);
        assert_eq!(
            is_transmutable(&src, &src, Assume { validity: true, ..Assume::default() }),
            Answer::Yes
        );
    }

    #[test]
    fn transmute_u8() {
        let bool = &Tree::bool();
        let u8 = &Tree::u8();
        for (src, dst, assume_validity, answer) in [
            (bool, u8, false, Answer::Yes),
            (bool, u8, true, Answer::Yes),
            (u8, bool, false, Answer::No(Reason::DstIsBitIncompatible)),
            (u8, bool, true, Answer::Yes),
        ] {
            assert_eq!(
                is_transmutable(
                    src,
                    dst,
                    Assume { validity: assume_validity, ..Assume::default() }
                ),
                answer
            );
        }
    }

    #[test]
    fn should_permit_validity_expansion_and_reject_contraction() {
        let b0 = layout::Tree::<Def, !, !>::byte(0);
        let b1 = layout::Tree::<Def, !, !>::byte(1);
        let b2 = layout::Tree::<Def, !, !>::byte(2);

        let alts = [b0, b1, b2];

        let into_layout = |alts: Vec<_>| {
            alts.into_iter()
                .fold(layout::Tree::<Def, !, !>::uninhabited(), layout::Tree::<Def, !, !>::or)
        };

        let into_set = |alts: Vec<_>| {
            #[cfg(feature = "rustc")]
            let mut set = rustc_data_structures::fx::FxIndexSet::default();
            #[cfg(not(feature = "rustc"))]
            let mut set = std::collections::HashSet::new();
            set.extend(alts);
            set
        };

        for src_alts in alts.clone().into_iter().powerset() {
            let src_layout = into_layout(src_alts.clone());
            let src_set = into_set(src_alts.clone());

            for dst_alts in alts.clone().into_iter().powerset().filter(|alts| !alts.is_empty()) {
                let dst_layout = into_layout(dst_alts.clone());
                let dst_set = into_set(dst_alts.clone());

                if src_set.is_subset(&dst_set) {
                    assert_eq!(
                        Answer::Yes,
                        is_transmutable(&src_layout, &dst_layout, Assume::default()),
                        "{:?} SHOULD be transmutable into {:?}",
                        src_layout,
                        dst_layout
                    );
                } else if !src_set.is_disjoint(&dst_set) {
                    assert_eq!(
                        Answer::Yes,
                        is_transmutable(
                            &src_layout,
                            &dst_layout,
                            Assume { validity: true, ..Assume::default() }
                        ),
                        "{:?} SHOULD be transmutable (assuming validity) into {:?}",
                        src_layout,
                        dst_layout
                    );
                } else {
                    assert_eq!(
                        Answer::No(Reason::DstIsBitIncompatible),
                        is_transmutable(&src_layout, &dst_layout, Assume::default()),
                        "{:?} should NOT be transmutable into {:?}",
                        src_layout,
                        dst_layout
                    );
                }
            }
        }
    }
}

mod uninit {
    use super::*;

    #[test]
    fn size() {
        let mu = Tree::uninit();
        let u8 = Tree::u8();

        for alignment in [false, true] {
            for lifetimes in [false, true] {
                for safety in [false, true] {
                    for validity in [false, true] {
                        let assume = Assume { alignment, lifetimes, safety, validity };

                        let want = if validity {
                            Answer::Yes
                        } else {
                            Answer::No(Reason::DstIsBitIncompatible)
                        };

                        assert_eq!(is_transmutable(&mu, &u8, assume), want, "assume: {assume:?}");
                        assert_eq!(
                            is_transmutable(&u8, &mu, assume),
                            Answer::Yes,
                            "assume: {assume:?}"
                        );
                    }
                }
            }
        }
    }
}

mod alt {
    use super::*;
    use crate::Answer;

    #[test]
    fn should_permit_identity_transmutation() {
        type Tree = layout::Tree<Def, !, !>;

        let x = Tree::Seq(vec![Tree::byte(0), Tree::byte(0)]);
        let y = Tree::Seq(vec![Tree::bool(), Tree::byte(1)]);
        let layout = Tree::Alt(vec![x, y]);

        let answer = crate::maybe_transmutable::MaybeTransmutableQuery::new(
            layout.clone(),
            layout.clone(),
            crate::Assume::default(),
            UltraMinimal::default(),
        )
        .answer();
        assert_eq!(answer, Answer::Yes, "layout:{:#?}", layout);
    }
}

mod union {
    use super::*;

    #[test]
    fn union() {
        let [a, b, c, d] = [0, 1, 2, 3];
        let s = Dfa::from_edges(a, d, &[(a, 0, b), (b, 0, d), (a, 1, c), (c, 1, d)]);

        let t = Dfa::from_edges(a, c, &[(a, 1, b), (b, 0, c)]);

        let mut ctr = 0;
        let new_state = || {
            let state = crate::layout::dfa::State(ctr);
            ctr += 1;
            state
        };

        let u = s.clone().union(t.clone(), new_state);

        let expected_u =
            Dfa::from_edges(b, a, &[(b, 0..=0, c), (b, 1..=1, d), (d, 0..=1, a), (c, 0..=0, a)]);

        assert_eq!(u, expected_u);

        assert_eq!(is_transmutable(&s, &u, Assume::default()), Answer::Yes);
        assert_eq!(is_transmutable(&t, &u, Assume::default()), Answer::Yes);
    }
}

mod char {
    use super::*;
    use crate::layout::tree::Endian;

    #[test]
    fn should_permit_valid_transmutation() {
        for order in [Endian::Big, Endian::Little] {
            use Answer::*;
            let char_layout = layout::Tree::<Def, !, !>::char(order);

            // `char`s can be in the following ranges:
            // - [0, 0xD7FF]
            // - [0xE000, 10FFFF]
            //
            // This loop synthesizes a singleton-validity type for the extremes
            // of each range, and for one past the end of the extremes of each
            // range.
            let no = No(Reason::DstIsBitIncompatible);
            for (src, answer) in [
                (0u32, Yes),
                (0xD7FF, Yes),
                (0xD800, no.clone()),
                (0xDFFF, no.clone()),
                (0xE000, Yes),
                (0x10FFFF, Yes),
                (0x110000, no.clone()),
                (0xFFFF0000, no.clone()),
                (0xFFFFFFFF, no),
            ] {
                let src_layout =
                    layout::tree::Tree::<Def, !, !>::from_big_endian(order, src.to_be_bytes());

                let a = is_transmutable(&src_layout, &char_layout, Assume::default());
                assert_eq!(a, answer, "endian:{order:?},\nsrc:{src:x}");
            }
        }
    }
}

mod nonzero {
    use super::*;
    use crate::{Answer, Reason};

    const NONZERO_BYTE_WIDTHS: [u64; 5] = [1, 2, 4, 8, 16];

    #[test]
    fn should_permit_identity_transmutation() {
        for width in NONZERO_BYTE_WIDTHS {
            let layout = layout::Tree::<Def, !, !>::nonzero(width);
            assert_eq!(is_transmutable(&layout, &layout, Assume::default()), Answer::Yes);
        }
    }

    #[test]
    fn should_permit_valid_transmutation() {
        for width in NONZERO_BYTE_WIDTHS {
            use Answer::*;

            let num = layout::Tree::<Def, !, !>::number(width);
            let nz = layout::Tree::<Def, !, !>::nonzero(width);

            let a = is_transmutable(&num, &nz, Assume::default());
            assert_eq!(a, No(Reason::DstIsBitIncompatible), "width:{width}");

            let a = is_transmutable(&nz, &num, Assume::default());
            assert_eq!(a, Yes, "width:{width}");
        }
    }
}

mod r#ref {
    use super::*;
    use crate::layout::Reference;

    #[test]
    fn should_permit_identity_transmutation() {
        type Tree = crate::layout::Tree<Def, usize, ()>;

        for validity in [false, true] {
            let layout = Tree::Seq(vec![
                Tree::byte(0x00),
                Tree::Ref(Reference {
                    region: 42,
                    is_mut: false,
                    referent: (),
                    referent_size: 0,
                    referent_align: 1,
                }),
            ]);

            let assume = Assume { validity, ..Assume::default() };

            let answer = crate::maybe_transmutable::MaybeTransmutableQuery::new(
                layout.clone(),
                layout,
                assume,
                UltraMinimal::default(),
            )
            .answer();
            assert_eq!(
                answer,
                Answer::If(Condition::IfAll(vec![
                    Condition::Transmutable { src: (), dst: () },
                    Condition::Outlives { long: 42, short: 42 },
                    Condition::Immutable { ty: () },
                ]))
            );
        }
    }
}

mod benches {
    use std::hint::black_box;

    use test::Bencher;

    use super::*;

    #[bench]
    fn bench_dfa_from_tree(b: &mut Bencher) {
        let num = Tree::number(8).prune(&|_| false);
        let num = black_box(num);

        b.iter(|| {
            let _ = black_box(Dfa::from_tree(num.clone()));
        })
    }

    #[bench]
    fn bench_transmute(b: &mut Bencher) {
        let num = Tree::number(8).prune(&|_| false);
        let dfa = black_box(Dfa::from_tree(num).unwrap());

        b.iter(|| {
            let answer = crate::maybe_transmutable::MaybeTransmutableQuery::new(
                dfa.clone(),
                dfa.clone(),
                Assume::default(),
                UltraMinimal::default(),
            )
            .answer();
            let answer = std::hint::black_box(answer);
            assert_eq!(answer, Answer::Yes);
        })
    }
}
