// Check that we don't leak stdlib implementation details through suggestions.
// Also check that the suggestion provided tries as hard as it can to see through local macros.
//
// FIXME(jieyouxu): this test is NOT run-rustfix because this test contains conflicting
// MaybeIncorrect suggestions:
//
// 1. `return ... ;`
// 2. `?`
//
// when the suggestions are applied to the same file, it becomes uncompilable.

// https://github.com/rust-lang/rust/issues/112007
pub fn bug_report<W: std::fmt::Write>(w: &mut W) -> std::fmt::Result {
    if true {
        writeln!(w, "`;?` here ->")?;
    } else {
        writeln!(w, "but not here")
        //~^ ERROR mismatched types
    }
    Ok(())
}

macro_rules! baz {
    ($w: expr) => {
        bar!($w)
    }
}

macro_rules! bar {
    ($w: expr) => {
        writeln!($w, "but not here")
        //~^ ERROR mismatched types
    }
}

pub fn foo<W: std::fmt::Write>(w: &mut W) -> std::fmt::Result {
    if true {
        writeln!(w, "`;?` here ->")?;
    } else {
        baz!(w)
    }
    Ok(())
}

pub fn main() {}
