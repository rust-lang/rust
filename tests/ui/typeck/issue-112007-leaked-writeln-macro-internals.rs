//@ run-rustfix
#![allow(dead_code)]

// https://github.com/rust-lang/rust/issues/112007
fn bug_report<W: std::fmt::Write>(w: &mut W) -> std::fmt::Result {
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

fn foo<W: std::fmt::Write>(w: &mut W) -> std::fmt::Result {
    if true {
        writeln!(w, "`;?` here ->")?;
    } else {
        baz!(w)
    }
    Ok(())
}

fn main() {}
