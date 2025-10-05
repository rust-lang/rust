//@ normalize-stderr: "pref: Align\([1-8] bytes\)" -> "pref: $$PREF_ALIGN"
//@ revisions: bit32 bit64
//@[bit32] only-32bit
//@[bit64] only-64bit
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
enum UninhabitedVariantSpace { //~ERROR: size: Size(15 bytes)
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
enum UninhabitedVariantUntagged { //~ERROR: size: Size(4 bytes)
    //~^ ERROR: abi: Scalar(Initialized
    A(i32),
    B(i32, !),
}

// A single-inhabited-variant enum may still be laid out with a tag,
// if that leads to a better niche for the same size layout.
// This enum uses the tagged representation, since the untagged representation would be
// the same size, but without a niche.
#[rustc_layout(size, abi)]
enum UninhabitedVariantUntaggedBigger { //~ERROR: size: Size(8 bytes)
    //~^ ERROR: abi: ScalarPair
    A(i32),
    B([u8; 5], !),
}

#[rustc_layout(size, abi)]
enum UninhabitedVariantWithNiche { //~ERROR: size: Size(2 bytes)
    //~^ERROR: abi: ScalarPair(Initialized { value: Int(I8, false), valid_range: 0..=1 }, Initialized { value: Int(I8, true), valid_range: 0..=255 })
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
enum AllUninhabitedVariants { //~ERROR: size: Size(2 bytes)
    //~^ERROR: abi: Memory
    A(i8, bool, !),
    B(u8, u8, !),
}

#[repr(align(2))]
struct AlignedNever(!);

// Tagged `(u8, i8)`
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
    //~^ERROR: abi: ScalarPair
    A(i16),
    B(i8, i8, i8, AlignedNever)
}

// This must not use tagged representation, since it's zero-sized.
#[rustc_layout(size, abi)]
enum AllUninhabitedVariantsAlignedZst { //~ERROR: size: Size(0 bytes)
    //~^ERROR: abi: Memory
    A(AlignedNever),
    B(AlignedNever),
}


#[repr(transparent)]
struct NPONever(&'static (), !);

#[repr(transparent)]
struct NPONeverI16(std::num::NonZero<i16>, !);

// All of these should be NPO-optimized, despite uninhabitedness of one or more variants
#[rustc_layout(abi)]
type NPONever1 = Result<NPONever, ()>;
//~^ ERROR: abi: Scalar(Initialized { value: Pointer

#[rustc_layout(abi)]
type NPONever2 = Result<(), NPONever>;
//~^ ERROR: abi: Scalar(Initialized { value: Pointer

#[rustc_layout(abi)]
type NPONever3 = Result<NPONever, !>;
//~^ ERROR: abi: Scalar(Initialized { value: Pointer

#[rustc_layout(abi)]
type NPONever4 = Result<!, NPONever>;
//~^ ERROR: abi: Scalar(Initialized { value: Pointer

#[rustc_layout(abi)]
type NPONever5 = Result<&'static (), !>;
//~^ ERROR: abi: Scalar(Initialized { value: Pointer

#[rustc_layout(abi)]
type NPONever6 = Result<!, &'static ()>;
//~^ ERROR: abi: Scalar(Initialized { value: Pointer

#[rustc_layout(abi)]
type NPONever7 = Result<std::num::NonZero<i16>, !>;
//~^ERROR: abi: Scalar(Initialized { value: Int(I16

#[rustc_layout(abi)]
type NPONever8 = Result<!, std::num::NonZero<i16>>;
//~^ERROR: abi: Scalar(Initialized { value: Int(I16

#[rustc_layout(abi)]
type NPONever9 = Result<NPONeverI16, !>;
//~^ERROR: abi: Scalar(Initialized { value: Int(I16

#[rustc_layout(abi)]
type NPONever10 = Result<!, NPONeverI16>;
//~^ERROR: abi: Scalar(Initialized { value: Int(I16

#[rustc_layout(abi)]
type NPONever11 = Result<NPONeverI16, ()>;
//~^ERROR: abi: Scalar(Initialized { value: Int(I16

#[rustc_layout(abi)]
type NPONever12 = Result<(), NPONeverI16>;
//~^ERROR: abi: Scalar(Initialized { value: Int(I16

#[rustc_layout(abi)]
enum NPONever13 { //~ERROR: abi: Scalar(Initialized { value: Pointer
    A(!, &'static (), !),
    B((), !, [u8; 0]),
}

#[rustc_layout(abi)]
enum NPONever14 { //~ERROR: abi: Scalar(Initialized { value: Pointer
    A(!, &'static (), !),
    B((), (), [u8; 0]),
}

#[rustc_layout(abi)]
enum NPONever15 { //~ERROR: abi: Scalar(Initialized { value: Pointer
    A((), &'static (), ()),
    B((), !, [u8; 0]),
}


// These are not guaranteed to be NPO-optimized
#[rustc_layout(abi)]
type NotNPONever1 = Result<NPONever, NPONever>;
//~^ERROR: abi: Scalar(Initialized { value: Int

#[rustc_layout(abi)]
type NotNPONever2 = Result<NPONever, AlignedNever>;
//~^ERROR: abi: Memory

#[rustc_layout(abi)]
type NotNPONever3 = Result<NPONever, &'static ()>;
//~^ERROR: abi: Scalar(

#[rustc_layout(abi)]
type NotNPONever4 = Result<&'static (), AlignedNever>;
//~^ERROR: abi: Scalar(
