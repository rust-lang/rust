// Tests to ensure we warn on repeated `#[repr(..)]` attributes.

#[repr(transparent, transparent)] //~ WARN `#[repr(..)]` attribute is specified more than once [repeated_reprs]
//~^ ERROR transparent struct cannot have other repr hints
#[repr(transparent)]
struct SeveralTransparentReprs(*mut u8);

#[repr(transparent)] //~ WARN `#[repr(..)]` attribute is specified more than once [repeated_reprs]
//~^ ERROR transparent struct cannot have other repr hints
#[repr(transparent)]
struct MultilineOnly(*mut u8);

#[repr(Rust, Rust)]
//~^ WARN `#[repr(..)]` attribute is specified more than once [repeated_reprs]
struct SeveralRustReprs(u8);

#[repr(C, C)]
//~^ WARN `#[repr(..)]` attribute is specified more than once [repeated_reprs]
#[repr(C, C, C)]
struct SeveralC(u8);

#[repr(u8, u8)] //~ WARN `#[repr(..)]` attribute is specified more than once [repeated_reprs]
//~^ ERROR conflicting representation hints
//~| WARN this was previously accepted
enum SeveralPrimitiveRerprs {
    Variant,
}

#[repr(C, C, u8)] //~ WARN `#[repr(..)]` attribute is specified more than once [repeated_reprs]
//~^ ERROR conflicting representation hints
//~| WARN this was previously accepted
#[repr(C, u8, u8)]
enum SeveralCAndPrims {
    Variant(u8),
}

#[repr(Rust, u8, u8)] //~ WARN `#[repr(..)]` attribute is specified more than once [repeated_reprs]
//~^ ERROR conflicting representation hints
//~^^ ERROR conflicting representation hints
//~| WARN this was previously accepted
enum RustAndPrimDisallowed {
    Variant(u8),
}

#[repr(u8, u8)] //~ WARN `#[repr(..)]` attribute is specified more than once [repeated_reprs]
//~^ ERROR conflicting representation hints
//~| WARN this was previously accepted
#[repr(u16)]
enum ConflictingPrimReprs {
    Variant,
}

#[repr(C, u8)]
//~^ ERROR conflicting representation hints
//~| WARN this was previously accepted
enum CWithIntsCausesFCW1 {
    A,
    B,
}

#[repr(C, C, u8, u8, u8)] //~ WARN `#[repr(..)]` attribute is specified more than once [repeated_reprs]
//~^ ERROR conflicting representation hints
//~| WARN this was previously accepted
enum CWithIntsCausesFCW2 {
    A,
    B,
}

fn main() {}
