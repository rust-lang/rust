use super::query_context::test::{Def, UltraMinimal};
use crate::maybe_transmutable::MaybeTransmutableQuery;
use crate::{layout, Answer, Reason, Set};
use itertools::Itertools;

mod bool {
    use super::*;

    #[test]
    fn should_permit_identity_transmutation_tree() {
        println!("{:?}", layout::Tree::<!, !>::bool());
        let answer = crate::maybe_transmutable::MaybeTransmutableQuery::new(
            layout::Tree::<Def, !>::bool(),
            layout::Tree::<Def, !>::bool(),
            (),
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
            (),
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
            let mut set = Set::default();
            #[cfg(not(feature = "rustc"))]
            let mut set = Set::new();
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
                            (),
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
                            (),
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
                            (),
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
