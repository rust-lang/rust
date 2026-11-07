//@ check-pass

//! [macro@m!] //~ WARN: unresolved link to `m`

//issue#126986

macro_rules! m {
    () => {};
}

fn main() {}
