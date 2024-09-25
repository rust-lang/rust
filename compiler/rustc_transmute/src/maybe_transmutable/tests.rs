use itertools::Itertools;

use super::query_context::test::{Def, UltraMinimal};
use crate::maybe_transmutable::MaybeTransmutableQuery;
use crate::{Reason, layout};

mod safety {
    use super::*;
    use crate::Answer;

    type Tree = layout::Tree<Def, !>;

    const DST_HAS_SAFETY_INVARIANTS: Answer<!> =
        Answer::No(crate::Reason::DstMayHaveSafetyInvariants);

    fn is_transmutable(src: &Tree, dst: &Tree, assume_safety: bool) -> crate::Answer<!> {
        let src = src.clone();
        let dst = dst.clone();
        // The only dimension of the transmutability analysis we want to test
        // here is the safety analysis. To ensure this, we disable all other
        // toggleable aspects of the transmutability analysis.
        let assume = crate::Assume {
            alignment: true,
            lifetimes: true,
            validity: true,
            safety: assume_safety,
        };
        crate::maybe_transmutable::MaybeTransmutableQuery::new(src, dst, assume, UltraMinimal)
            .answer()
    }

    #[test]
    fn src_safe_dst_safe() {
        let src = Tree::Def(Def::NoSafetyInvariants).then(Tree::u8());
        let dst = Tree::Def(Def::NoSafetyInvariants).then(Tree::u8());
        assert_eq!(is_transmutable(&src, &dst, false), Answer::Yes);
        assert_eq!(is_transmutable(&src, &dst, true), Answer::Yes);
    }

    #[test]
    fn src_safe_dst_unsafe() {
        let src = Tree::Def(Def::NoSafetyInvariants).then(Tree::u8());
        let dst = Tree::Def(Def::HasSafetyInvariants).then(Tree::u8());
        assert_eq!(is_transmutable(&src, &dst, false), DST_HAS_SAFETY_INVARIANTS);
        assert_eq!(is_transmutable(&src, &dst, true), Answer::Yes);
    }

    #[test]
    fn src_unsafe_dst_safe() {
        let src = Tree::Def(Def::HasSafetyInvariants).then(Tree::u8());
        let dst = Tree::Def(Def::NoSafetyInvariants).then(Tree::u8());
        assert_eq!(is_transmutable(&src, &dst, false), Answer::Yes);
        assert_eq!(is_transmutable(&src, &dst, true), Answer::Yes);
    }

    #[test]
    fn src_unsafe_dst_unsafe() {
        let src = Tree::Def(Def::HasSafetyInvariants).then(Tree::u8());
        let dst = Tree::Def(Def::HasSafetyInvariants).then(Tree::u8());
        assert_eq!(is_transmutable(&src, &dst, false), DST_HAS_SAFETY_INVARIANTS);
        assert_eq!(is_transmutable(&src, &dst, true), Answer::Yes);
    }
}

mod bool {
    use super::*;
    use crate::Answer;

    #[test]
    fn should_permit_identity_transmutation_tree() {
        let answer = crate::maybe_transmutable::MaybeTransmutableQuery::new(
            layout::Tree::<Def, !>::bool(),
            layout::Tree::<Def, !>::bool(),
            crate::Assume { alignment: false, lifetimes: false, validity: true, safety: false },
            UltraMinimal,
        )
        .answer();
        assert_eq!(answer, Answer::Yes);
    }

    #[test]
    fn should_permit_identity_transmutation_dfa() {
        let answer = crate::maybe_transmutable::MaybeTransmutableQuery::new(
            layout::Dfa::<!>::bool(),
            layout::Dfa::<!>::bool(),
            crate::Assume { alignment: false, lifetimes: false, validity: true, safety: false },
            UltraMinimal,
        )
        .answer();
        assert_eq!(answer, Answer::Yes);
    }

    #[test]
    fn should_permit_validity_expansion_and_reject_contraction() {
        let un = layout::Tree::<Def, !>::uninhabited();
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
                        MaybeTransmutableQuery::new(
                            src_layout.clone(),
                            dst_layout.clone(),
                            crate::Assume { validity: false, ..crate::Assume::default() },
                            UltraMinimal,
                        )
                        .answer(),
                        "{:?} SHOULD be transmutable into {:?}",
                        src_layout,
                        dst_layout
                    );
                } else if !src_set.is_disjoint(&dst_set) {
                    assert_eq!(
                        Answer::Yes,
                        MaybeTransmutableQuery::new(
                            src_layout.clone(),
                            dst_layout.clone(),
                            crate::Assume { validity: true, ..crate::Assume::default() },
                            UltraMinimal,
                        )
                        .answer(),
                        "{:?} SHOULD be transmutable (assuming validity) into {:?}",
                        src_layout,
                        dst_layout
                    );
                } else {
                    assert_eq!(
                        Answer::No(Reason::DstIsBitIncompatible),
                        MaybeTransmutableQuery::new(
                            src_layout.clone(),
                            dst_layout.clone(),
                            crate::Assume { validity: false, ..crate::Assume::default() },
                            UltraMinimal,
                        )
                        .answer(),
                        "{:?} should NOT be transmutable into {:?}",
                        src_layout,
                        dst_layout
                    );
                }
            }
        }
    }
}
