//@ normalize-stderr: "pref: Align\([1-8] bytes\)" -> "pref: $$PREF_ALIGN"
//! Various enum layout tests.

#![feature(rustc_attrs)]
#![feature(never_type)]
#![crate_type = "lib"]

#[rustc_layout(align)]
enum UninhabitedVariantAlign { //~ERROR: abi: Align(2 bytes)
    A([u8; 32]),
    B([u16; 0], !), // make sure alignment in uninhabited fields is respected
}

#[rustc_layout(size)]
enum UninhabitedVariantSpace { //~ERROR: size: Size(16 bytes)
    A,
    B([u8; 15], !), // make sure there is space being reserved for this field.
}

#[rustc_layout(abi)]
enum ScalarPairDifferingSign { //~ERROR: abi: ScalarPair
    A(u8),
    B(i8),
}

// Enums with only a single inhabited variant can be laid out as just that variant,
// if the uninhabited variants are all "absent" (only have 1-ZST fields)
#[rustc_layout(size, abi)]
enum AbsentVariantUntagged { //~ERROR: size: Size(4 bytes)
    //~^ ERROR: abi: Scalar(Initialized
    A(i32),
    B((), !),
}

// Even if uninhabited variants are not absent, the enum can still be laid out without
// a tag.
#[rustc_layout(size, abi)]
enum UninhabitedVariantUntagged { //~ERROR: size: Size(8 bytes)
    //~^ ERROR: abi: ScalarPair(Initialized
    A(i32),
    B(i32, !),
}

// A single-inhabited-variant enum may still be laid out with a tag,
// if that leads to a better niche for the same size layout.
// This enum uses the tagged representation, since the untagged representation would be
// the same size, but without a niche.
#[rustc_layout(size, abi)]
enum UninhabitedVariantUntaggedBigger { //~ERROR: size: Size(8 bytes)
    //~^ ERROR: abi: Memory
    A(i32),
    B([u8; 5], !),
}

#[rustc_layout(size, abi)]
enum UninhabitedVariantWithNiche { //~ERROR: size: Size(3 bytes)
    //~^ERROR: abi: Memory
    A(i8, bool),
    B(u8, u8, !),
}

#[rustc_layout(debug)]
enum UninhabitedVariantLargeWithNiche {
    //~^ ERROR: layout_of
    //~| ERROR: size: Size(3 bytes)
    //~| ERROR: backend_repr: Memory
    //~| ERROR: valid_range: 0..=0
    // Should use the tagged representation, since that gives a 255-slot niche,
    // instead of a 254-slot niche if it used the niche-filling representation on the `bool`
    A(i8, bool),
    B(u8, u8, u8, !),
}

// This uses the tagged layout, but since all variants are uninhabited, none of them store the tag,
// so we only need space for the fields, and the abi is Memory.
#[rustc_layout(size, abi)]
enum AllUninhabitedVariants { //~ERROR: size: Size(3 bytes)
    //~^ERROR: abi: Memory
    A(i8, bool, !),
    B(u8, u8, !),
}

#[repr(align(2))]
struct AlignedNever(!);

// Tagged `(i8, padding)`
#[rustc_layout(size, abi)]
enum AlignedI8 { //~ERROR: size: Size(2 bytes)
    //~^ERROR: abi: Memory
    A(i8),
    B(AlignedNever)
}

// Tagged `(u8, i8, padding, padding)`
#[rustc_layout(size, abi)]
enum TaggedI8 { //~ERROR: size: Size(4 bytes)
    //~^ERROR: abi: Memory
    A(i8),
    B(i8, i8, i8, AlignedNever)
}


// Tagged `(u16, i16)`
#[rustc_layout(size, abi)]
enum TaggedI16 { //~ERROR: size: Size(4 bytes)
    //~^ERROR: abi: Memory
    A(i16),
    B(i8, i8, i8, AlignedNever)
}

// This must not use tagged representation, since it's zero-sized.
#[rustc_layout(size, abi)]
enum AllUninhabitedVariantsAlignedZst { //~ERROR: size: Size(2 bytes)
    //~^ERROR: abi: Scalar
    A(AlignedNever),
    B(AlignedNever),
}
