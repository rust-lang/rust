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
        crate::maybe_transmutable::MaybeTransmutableQuery::new(src, dst, assume, UltraMinimal)
            .answer()
    }
}

impl Representation for Dfa {
    fn is_transmutable(src: Self, dst: Self, assume: Assume) -> Answer<!> {
        crate::maybe_transmutable::MaybeTransmutableQuery::new(src, dst, assume, UltraMinimal)
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
            let mut set = crate::Set::default();
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
