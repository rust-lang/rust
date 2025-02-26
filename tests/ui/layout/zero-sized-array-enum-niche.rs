//@ normalize-stderr: "pref: Align\([1-8] bytes\)" -> "pref: $$PREF_ALIGN"
//@ normalize-stderr: "randomization_seed: \d+" -> "randomization_seed: $$SEED"
#![crate_type = "lib"]
#![feature(rustc_attrs)]

// Various tests around the behavior of zero-sized arrays and
// enum niches, especially that they have coherent size and alignment.

// The original problem in #99836 came from ndarray's `TryFrom` for
// `SliceInfo<[SliceInfoElem; 0], Din, Dout>`, where that returns
// `Result<Self, ShapeError>` ~= `Result<AlignedZST, TypeWithNiche>`.
// This is a close enough approximation:
#[rustc_layout(debug)]
type AlignedResult = Result<[u32; 0], bool>; //~ ERROR: layout_of
// The bug gave that size 1 with align 4, but the size should also be 4.
// It was also using the bool niche for the enum tag, which is fine, but
// after the fix, layout decides to use a direct tagged repr instead.

// Here's another case with multiple ZST alignments, where we should
// get the maximal alignment and matching size.
#[rustc_layout(debug)]
enum MultipleAlignments { //~ ERROR: layout_of
    Align2([u16; 0]),
    Align4([u32; 0]),
    Niche(bool),
}

// Tagged repr is clever enough to grow tags to fill any padding, e.g.:
// 1.   `T_FF` (one byte of Tag, one byte of padding, two bytes of align=2 Field)
//   -> `TTFF` (Tag has expanded to two bytes, i.e. like `#[repr(u16)]`)
// 2.    `TFF` (one byte of Tag, two bytes of align=1 Field)
//   -> Tag has no room to expand!
//   (this outcome can be forced onto 1. by wrapping Field in `Packed<...>`)
#[repr(packed)]
struct Packed<T>(T);

#[rustc_layout(debug)]
type NicheLosesToTagged = Result<[u32; 0], Packed<std::num::NonZero<u16>>>; //~ ERROR: layout_of
// Should get tag_encoding: Direct, size == align == 4.

#[repr(u16)]
enum U16IsZero { _Zero = 0 }

#[rustc_layout(debug)]
type NicheWinsOverTagged = Result<[u32; 0], Packed<U16IsZero>>; //~ ERROR: layout_of
// Should get tag_encoding: Niche, size == align == 4.
