// check-fail
// This was originally causing an ICE in `rustc_transmute::maybe_transmutable`
#![crate_type="lib"]
#[repr(C, packed)] //~ ERROR attribute should be applied to a struct or union
enum V0usize {
    V,
}
