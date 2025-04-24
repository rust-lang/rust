extern crate test;

use itertools::Itertools;

use super::query_context::test::{Def, UltraMinimal};
use crate::{Answer, Assume, Reason, layout};

type Tree = layout::Tree<Def, !>;
type Dfa = layout::Dfa<!>;

trait Representation {
    fn is_transmutable(src: Self, dst: Self, assume: Assume) -> Answer<!>;
}

impl Representation for Tree {
    fn is_transmutable(src: Self, dst: Self, assume: Assume) -> Answer<!> {
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
    fn is_transmutable(src: Self, dst: Self, assume: Assume) -> Answer<!> {
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
) -> crate::Answer<!> {
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

    const DST_HAS_SAFETY_INVARIANTS: Answer<!> =
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
        let b0 = layout::Tree::<Def, !>::from_bits(0);
        let b1 = layout::Tree::<Def, !>::from_bits(1);
        let b2 = layout::Tree::<Def, !>::from_bits(2);

        let alts = [b0, b1, b2];

        let into_layout = |alts: Vec<_>| {
            alts.into_iter().fold(layout::Tree::<Def, !>::uninhabited(), layout::Tree::<Def, !>::or)
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
        type Tree = layout::Tree<Def, !>;

        let x = Tree::Seq(vec![Tree::from_bits(0), Tree::from_bits(0)]);
        let y = Tree::Seq(vec![Tree::bool(), Tree::from_bits(1)]);
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
            Dfa::from_edges(b, a, &[(b, 0, c), (b, 1, d), (d, 1, a), (d, 0, a), (c, 0, a)]);

        assert_eq!(u, expected_u);

        assert_eq!(is_transmutable(&s, &u, Assume::default()), Answer::Yes);
        assert_eq!(is_transmutable(&t, &u, Assume::default()), Answer::Yes);
    }
}

mod r#ref {
    use super::*;

    #[test]
    fn should_permit_identity_transmutation() {
        type Tree = crate::layout::Tree<Def, [(); 1]>;

        let layout = Tree::Seq(vec![Tree::from_bits(0), Tree::Ref([()])]);

        let answer = crate::maybe_transmutable::MaybeTransmutableQuery::new(
            layout.clone(),
            layout,
            Assume::default(),
            UltraMinimal::default(),
        )
        .answer();
        assert_eq!(answer, Answer::If(crate::Condition::IfTransmutable { src: [()], dst: [()] }));
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
