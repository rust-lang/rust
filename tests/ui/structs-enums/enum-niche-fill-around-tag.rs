//@ revisions: old future
//@ run-pass
//@ [old] edition:2024
//@ [future] edition:future
//@ [future] compile-flags: -Z unstable-options

#![allow(dead_code)]
#![cfg_attr(future, feature(offset_of_enum))]

#[cfg(future)]
use std::mem::{align_of, offset_of};
use std::mem::size_of;
use std::num::NonZero;

enum Inner {
    A(u8, u32),
    B(u32),
}

enum Outer {
    A(Inner),
    B(u32, u8),
}

struct WithPadding {
    a: bool,
    b: bool,
    c: u32,
}

enum RepackPadding {
    A(u32, u8),
    B(WithPadding),
}

struct RepackedBase {
    int32: u32,
    boolean: bool,
}

enum RepackedAroundNiche {
    A(RepackedBase),
    B(u16, u32, u8),
}

enum NestedRepackedAroundNiche {
    A(RepackedAroundNiche),
    B(u16, u32, u8),
}

struct RepackedPair {
    field2: u16,
    field3: u8,
}

enum RepackedNestedAggregate {
    A(u16, RepackedPair),
    B(NestedRepackedAroundNiche),
}

struct FittingNichePayload {
    word: u32,
    flag: bool,
}

// The first and last max-sized niche candidates cannot encode all variants.
// The middle candidate can, so the enum layout has to keep looking.
enum ChooseFittingNiche {
    SmallNicheFront(u16, u32, NonZero<u8>),
    WideNiche(FittingNichePayload),
    SmallNicheBack(u16, u32, NonZero<u8>),
    V0,
    V1,
    V2,
}

fn main() {
    assert_eq!(size_of::<Inner>(), 8);

    #[cfg(old)]
    {
        assert_eq!(size_of::<Outer>(), 12);
        assert_eq!(size_of::<RepackPadding>(), 12);
    }

    #[cfg(future)]
    {
        assert_eq!(size_of::<Outer>(), 8);
        assert_eq!(size_of::<RepackPadding>(), 8);

        assert_eq!(size_of::<RepackedBase>(), 8);
        assert_eq!(align_of::<RepackedBase>(), 4);
        assert_eq!(offset_of!(RepackedBase, int32), 0);
        assert_eq!(offset_of!(RepackedBase, boolean), 4);

        assert_eq!(size_of::<RepackedAroundNiche>(), 8);
        assert_eq!(align_of::<RepackedAroundNiche>(), 4);
        assert_eq!(size_of::<Option<RepackedAroundNiche>>(), 8);
        assert_eq!(offset_of!(RepackedAroundNiche, A.0), 0);
        assert_eq!(offset_of!(RepackedAroundNiche, B.1), 0);
        assert_eq!(offset_of!(RepackedAroundNiche, B.2), 5);
        assert_eq!(offset_of!(RepackedAroundNiche, B.0), 6);

        assert_eq!(size_of::<NestedRepackedAroundNiche>(), 8);
        assert_eq!(align_of::<NestedRepackedAroundNiche>(), 4);
        assert_eq!(size_of::<Option<NestedRepackedAroundNiche>>(), 8);
        assert_eq!(offset_of!(NestedRepackedAroundNiche, A.0), 0);
        assert_eq!(offset_of!(NestedRepackedAroundNiche, B.1), 0);
        assert_eq!(offset_of!(NestedRepackedAroundNiche, B.2), 5);
        assert_eq!(offset_of!(NestedRepackedAroundNiche, B.0), 6);

        assert_eq!(size_of::<RepackedPair>(), 4);
        assert_eq!(align_of::<RepackedPair>(), 2);
        assert_eq!(offset_of!(RepackedPair, field2), 0);
        assert_eq!(offset_of!(RepackedPair, field3), 2);

        assert_eq!(size_of::<RepackedNestedAggregate>(), 8);
        assert_eq!(align_of::<RepackedNestedAggregate>(), 4);
        assert_eq!(size_of::<Option<RepackedNestedAggregate>>(), 8);
        assert_eq!(offset_of!(RepackedNestedAggregate, B.0), 0);
        assert_eq!(offset_of!(RepackedNestedAggregate, A.1), 0);
        assert_eq!(offset_of!(RepackedNestedAggregate, A.0), 6);

        assert_eq!(size_of::<FittingNichePayload>(), 8);
        assert_eq!(align_of::<FittingNichePayload>(), 4);
        assert_eq!(offset_of!(FittingNichePayload, word), 0);
        assert_eq!(offset_of!(FittingNichePayload, flag), 4);

        assert_eq!(size_of::<ChooseFittingNiche>(), 8);
        assert_eq!(align_of::<ChooseFittingNiche>(), 4);
        assert_eq!(size_of::<Option<ChooseFittingNiche>>(), 8);
        assert_eq!(offset_of!(ChooseFittingNiche, WideNiche.0), 0);
        assert_eq!(offset_of!(ChooseFittingNiche, SmallNicheFront.1), 0);
        assert_eq!(offset_of!(ChooseFittingNiche, SmallNicheFront.2), 5);
        assert_eq!(offset_of!(ChooseFittingNiche, SmallNicheFront.0), 6);
        assert_eq!(offset_of!(ChooseFittingNiche, SmallNicheBack.1), 0);
        assert_eq!(offset_of!(ChooseFittingNiche, SmallNicheBack.2), 5);
        assert_eq!(offset_of!(ChooseFittingNiche, SmallNicheBack.0), 6);
    }
}
