// Tests that repeated attributes are allowed.

#[repr(transparent, transparent)]
#[repr(transparent)]
struct SeveralTransparentReprs(*mut u8);

#[repr(transparent)]
#[repr(transparent)]
struct MultilineOnly(*mut u8);

#[repr(Rust, Rust)]
struct SeveralRustReprs(u8);

#[repr(C, C)]
#[repr(C, C, C)]
struct SeveralC(u8);

#[repr(u8, u8)]
enum SeveralPrimitiveRerprs {
    Variant,
}

#[repr(C, C, u8)]
#[repr(C, u8, u8)]
enum SeveralCAndPrims {
    Variant(u8),
}

#[repr(Rust, u8, u8)]
//~^ ERROR conflicting representation hints [E0566]
enum RustAndPrimDisallowed {
    Variant(u8),
}

fn main() {}
