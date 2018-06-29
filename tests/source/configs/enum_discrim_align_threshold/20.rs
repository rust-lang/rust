// rustfmt-enum_discrim_align_threshold: 20

enum Standard {
    A = 1,
    Bcdef = 2,
}

enum Mixed {
    ThisIsAFairlyLongEnumVariantWithoutDiscrim,
    A = 1,
    ThisIsAFairlyLongEnumVariantWithoutDiscrim2,
    Bcdef = 2,
}

enum TooLong {
    ThisOneHasDiscrimAaaaaaaaaaaaaaaaaaaaaaaaaaaa = 10,
    A = 1,
    Bcdef = 2,
}

// Live specimen from #1686
enum LongWithSmallDiff {
    SceneColorimetryEstimates = 0x73636F65,
    SceneAppearanceEstimates = 0x73617065,
    FocalPlaneColorimetryEstimates = 0x66706365,
    ReflectionHardcopyOriginalColorimetry = 0x72686F63,
    ReflectionPrintOutputColorimetry = 0x72706F63,
}