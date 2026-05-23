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

#[repr(u8, u8)]
//~^ ERROR conflicting representation hints [E0566]
//~| WARN: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
#[repr(u16)]
enum ConflictingPrimReprs {
    Variant,
}

fn main() {}
