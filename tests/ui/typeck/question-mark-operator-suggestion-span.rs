// Check that we don't construct a span for `?` suggestions that point into non-local macros
// like into the stdlib where the user has no control over.
//
// FIXME(jieyouxu): this test is currently NOT run-rustfix because there are conflicting
// MaybeIncorrect suggestions:
//
// 1. adding `return ... ;`, and
// 2. adding `?`.
//
// When rustfix puts those together, the fixed file now contains uncompilable code.

#![crate_type = "lib"]

pub fn bug_report<W: std::fmt::Write>(w: &mut W) -> std::fmt::Result {
    if true {
        writeln!(w, "`;?` here ->")?;
    } else {
        writeln!(w, "but not here")
        //~^ ERROR mismatched types
    }
    Ok(())
}
