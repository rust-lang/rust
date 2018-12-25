#![feature(repr_simd, platform_intrinsics)]

#[repr(simd)]
struct A(i16, i16, i16, i16, i16, i16, i16, i16);
#[repr(simd)]
struct B(i16, i16, i16, i16, i16, i16, i16, i16);

// each intrinsic definition has to use the same nominal type for any
// vector structure throughout that declaration (i.e., every instance
// of i16x8 in each `fn ...;` needs to be either A or B)

extern "platform-intrinsic" {
    fn x86_mm_adds_epi16(x: A, y: A) -> B;
    //~^ ERROR intrinsic return value has wrong type: found `B`, expected `A`
    fn x86_mm_subs_epi16(x: A, y: B) -> A;
    //~^ ERROR intrinsic argument 2 has wrong type: found `B`, expected `A`

    // ok:
    fn x86_mm_max_epi16(x: B, y: B) -> B;
    fn x86_mm_min_epi16(x: A, y: A) -> A;
}

fn main() {}
