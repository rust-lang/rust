//@ compile-flags: --crate-type=lib
//@ check-pass
//@ edition: 2024
#![feature(const_raw_ptr_comparison)]
#![feature(fn_align)]
// Generally:
// For any `Some` return, `None` would also be valid, unless otherwise noted.
// For any `None` return, only `None` is valid, unless otherwise noted.

macro_rules! do_test {
    ($a:expr, $b:expr, $expected:pat) => {
        const _: () = {
            let a: *const _ = $a;
            let b: *const _ = $b;
            assert!(matches!(<*const u8>::guaranteed_eq(a.cast(), b.cast()), $expected));
        };
    };
}

#[repr(align(2))]
struct T(#[allow(unused)] u16);

#[repr(align(2))]
struct AlignedZst;

static A: T = T(42);
static B: T = T(42);
static mut MUT_STATIC: T = T(42);
static ZST: () = ();
static ALIGNED_ZST: AlignedZst = AlignedZst;
static LARGE_WORD_ALIGNED: [usize; 2] = [0, 1];
static mut MUT_LARGE_WORD_ALIGNED: [usize; 2] = [0, 1];

const FN_PTR: *const () = {
    fn foo() {}
    unsafe { std::mem::transmute(foo as fn()) }
};

const ALIGNED_FN_PTR: *const () = {
    #[rustc_align(2)]
    fn aligned_foo() {}
    unsafe { std::mem::transmute(aligned_foo as fn()) }
};

trait Trait {
    #[allow(unused)]
    fn method(&self) -> u8;
}
impl Trait for u32 {
    fn method(&self) -> u8 { 1 }
}
impl Trait for i32 {
    fn method(&self) -> u8 { 2 }
}

const VTABLE_PTR_1: *const () = {
    let [_data, vtable] = unsafe {
        std::mem::transmute::<&dyn Trait, [*const (); 2]>(&42_u32 as &dyn Trait)
    };
    vtable
};
const VTABLE_PTR_2: *const () = {
    let [_data, vtable] = unsafe {
        std::mem::transmute::<&dyn Trait, [*const (); 2]>(&42_i32 as &dyn Trait)
    };
    vtable
};

// Cannot be `None`: `is_null` is stable with strong guarantees about integer-valued pointers.
do_test!(0 as *const u8, 0 as *const u8, Some(true));
do_test!(0 as *const u8, 1 as *const u8, Some(false));

// Integer-valued pointers can always be compared.
do_test!(1 as *const u8, 1 as *const u8, Some(true));
do_test!(1 as *const u8, 2 as *const u8, Some(false));

// Cannot be `None`: `static`s' addresses, references, (and within and one-past-the-end of those),
// and `fn` pointers cannot be null, and `is_null` is stable with strong guarantees, and
// `is_null` is implemented using `guaranteed_cmp`.
do_test!(&A, 0 as *const u8, Some(false));
do_test!((&raw const A).cast::<u8>().wrapping_add(1), 0 as *const u8, Some(false));
do_test!((&raw const A).wrapping_add(1), 0 as *const u8, Some(false));
do_test!(&ZST, 0 as *const u8, Some(false));
do_test!(&(), 0 as *const u8, Some(false));
do_test!(const { &() }, 0 as *const u8, Some(false));
do_test!(FN_PTR, 0 as *const u8, Some(false));

// This pointer is out-of-bounds, but still cannot be equal to 0 because of alignment.
do_test!((&raw const A).cast::<u8>().wrapping_add(size_of::<T>() + 1), 0 as *const u8, Some(false));

// aside from 0, these pointers might end up pretty much anywhere.
do_test!(&A, align_of::<T>() as *const u8, None);
do_test!((&raw const A).wrapping_byte_add(1), (align_of::<T>() + 1) as *const u8, None);

// except that they must still be aligned
do_test!(&A, 1 as *const u8, Some(false));
do_test!((&raw const A).wrapping_byte_add(1), align_of::<T>() as *const u8, Some(false));

// If `ptr.wrapping_sub(int)` cannot be null (because it is in-bounds or one-past-the-end of
// `ptr`'s allocation, or because it is misaligned from `ptr`'s allocation), then we know that
// `ptr != int`, even if `ptr` itself is out-of-bounds or one-past-the-end of its allocation.
do_test!((&raw const A).wrapping_byte_add(1), 1 as *const u8, Some(false));
do_test!((&raw const A).wrapping_byte_add(2), 2 as *const u8, Some(false));
do_test!((&raw const A).wrapping_byte_add(3), 1 as *const u8, Some(false));
do_test!((&raw const ZST).wrapping_byte_add(1), 1 as *const u8, Some(false));
do_test!(VTABLE_PTR_1.wrapping_byte_add(1), 1 as *const u8, Some(false));
do_test!(FN_PTR.wrapping_byte_add(1), 1 as *const u8, Some(false));
do_test!(&A, size_of::<T>().wrapping_neg() as *const u8, Some(false));
do_test!(&LARGE_WORD_ALIGNED, size_of::<usize>().wrapping_neg() as *const u8, Some(false));
// (`ptr - int != 0` due to misalignment)
do_test!((&raw const A).wrapping_byte_add(2), 1 as *const u8, Some(false));
do_test!((&raw const ALIGNED_ZST).wrapping_byte_add(2), 1 as *const u8, Some(false));

// When pointers go out-of-bounds, they *might* become null, so these comparions cannot work.
do_test!((&raw const A).wrapping_add(2), 0 as *const u8, None);
do_test!((&raw const A).wrapping_sub(1), 0 as *const u8, None);

// Statics cannot be duplicated
do_test!(&A, &A, Some(true));

// Two non-ZST statics cannot have the same address
do_test!(&A, &B, Some(false));
do_test!(&A, &raw const MUT_STATIC, Some(false));

// One-past-the-end of one static can be equal to the address of another static.
do_test!(&A, (&raw const B).wrapping_add(1), None);

// Cannot know if ZST static is at the same address with anything non-null (if alignment allows).
do_test!(&A, &ZST, None);
do_test!(&A, &ALIGNED_ZST, None);

// Unclear if ZST statics can be placed "in the middle of" non-ZST statics.
// For now, we conservatively say they could, and return None here.
do_test!(&ZST, (&raw const A).wrapping_byte_add(1), None);

// As per https://doc.rust-lang.org/nightly/reference/items/static-items.html#r-items.static.storage-disjointness
// immutable statics are allowed to overlap with const items and promoteds.
do_test!(&A, &T(42), None);
do_test!(&A, const { &T(42) }, None);
do_test!(&A, { const X: T = T(42); &X }, None);

// These could return Some(false), since only immutable statics can overlap with const items
// and promoteds.
do_test!(&raw const MUT_STATIC, &T(42), None);
do_test!(&raw const MUT_STATIC, const { &T(42) }, None);
do_test!(&raw const MUT_STATIC, { const X: T = T(42); &X }, None);

// An odd offset from a 2-aligned allocation can never be equal to an even offset from a
// 2-aligned allocation, even if the offsets are out-of-bounds.
do_test!(&A, (&raw const B).wrapping_byte_add(1), Some(false));
do_test!(&A, (&raw const B).wrapping_byte_add(5), Some(false));
do_test!(&A, (&raw const ALIGNED_ZST).wrapping_byte_add(1), Some(false));
do_test!(&ALIGNED_ZST, (&raw const A).wrapping_byte_add(1), Some(false));
do_test!(&A, (&T(42) as *const T).wrapping_byte_add(1), Some(false));
do_test!(&A, (const { &T(42) } as *const T).wrapping_byte_add(1), Some(false));
do_test!(&A, ({ const X: T = T(42); &X } as *const T).wrapping_byte_add(1), Some(false));

// We could return `Some(false)` for these, as pointers to different statics can never be equal if
// that would require the statics to overlap, even if the pointers themselves are offset out of
// bounds or one-past-the-end. We currently only check strictly in-bounds pointers when comparing
// pointers to different statics, however.
do_test!((&raw const A).wrapping_add(1), (&raw const B).wrapping_add(1), None);
do_test!(
    (&raw const LARGE_WORD_ALIGNED).cast::<usize>().wrapping_add(2),
    (&raw const MUT_LARGE_WORD_ALIGNED).cast::<usize>().wrapping_add(1),
    None
);

// Pointers into the same static are equal if and only if their offset is the same,
// even if either is out-of-bounds.
do_test!(&A, &A, Some(true));
do_test!(&A, &A.0, Some(true));
do_test!(&A, (&raw const A).wrapping_byte_add(1), Some(false));
do_test!(&A, (&raw const A).wrapping_byte_add(2), Some(false));
do_test!(&A, (&raw const A).wrapping_byte_add(51), Some(false));
do_test!((&raw const A).wrapping_byte_add(51), (&raw const A).wrapping_byte_add(51), Some(true));

// Pointers to the same fn may be unequal, since `fn`s can be duplicated.
do_test!(FN_PTR, FN_PTR, None);
do_test!(ALIGNED_FN_PTR, ALIGNED_FN_PTR, None);

// Pointers to different fns may be equal, since `fn`s can be deduplicated.
do_test!(FN_PTR, ALIGNED_FN_PTR, None);

// Pointers to the same vtable may be unequal, since vtables can be duplicated.
do_test!(VTABLE_PTR_1, VTABLE_PTR_1, None);

// Pointers to different vtables may be equal, since vtables can be deduplicated.
do_test!(VTABLE_PTR_1, VTABLE_PTR_2, None);

// Function pointers to aligned function allocations are not necessarily actually aligned,
// due to platform-specific semantics.
// See https://github.com/rust-lang/rust/issues/144661
// FIXME: This could return `Some` on platforms where function pointers' addresses actually
// correspond to function addresses including alignment, or on platforms where all functions
// are aligned to some amount (e.g. ARM where a32 function pointers are at least 4-aligned,
// and t32 function pointers are 2-aligned-offset-by-1).
do_test!(ALIGNED_FN_PTR, ALIGNED_FN_PTR.wrapping_byte_offset(1), None);

// Conservatively say we don't know.
do_test!(FN_PTR, VTABLE_PTR_1, None);
do_test!((&raw const LARGE_WORD_ALIGNED).cast::<usize>().wrapping_add(1), VTABLE_PTR_1, None);
do_test!((&raw const MUT_LARGE_WORD_ALIGNED).cast::<usize>().wrapping_add(1), VTABLE_PTR_1, None);
do_test!((&raw const LARGE_WORD_ALIGNED).cast::<usize>().wrapping_add(1), FN_PTR, None);
do_test!((&raw const MUT_LARGE_WORD_ALIGNED).cast::<usize>().wrapping_add(1), FN_PTR, None);
