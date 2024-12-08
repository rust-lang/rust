// issue: rust-lang/rust#66757
//
// See also: `tests/ui/never_type/from_infer_breaking_with_unit_fallback.rs`.
//
//@ revisions: unit never
//@ check-pass
#![allow(internal_features)]
#![feature(rustc_attrs, never_type)]
#![cfg_attr(unit, rustc_never_type_options(fallback = "unit"))]
#![cfg_attr(never, rustc_never_type_options(fallback = "never"))]

type Infallible = !;

struct E;

impl From<Infallible> for E {
    fn from(_: Infallible) -> E {
        E
    }
}

fn u32_try_from(x: u32) -> Result<u32, Infallible> {
    Ok(x)
}

fn _f() -> Result<(), E> {
    // In an old attempt to make `Infallible = !` this caused a problem.
    //
    // Because at the time the code desugared to
    //
    //   match u32::try_from(1u32) {
    //       Ok(x) => x, Err(e) => return Err(E::from(e))
    //   }
    //
    // With `Infallible = !`, `e: !` but with fallback to `()`, `e` in `E::from(e)` decayed to `()`
    // causing an error.
    //
    // This does not happen with `Infallible = !`.
    // And also does not happen with the newer `?` desugaring that does not pass `e` by value.
    // (instead we only pass `Result<!, Error>` (where `Error = !` in this case) which does not get
    // the implicit coercion and thus does not decay even with fallback to unit)
    u32_try_from(1u32)?;
    Ok(())
}

fn main() {}
