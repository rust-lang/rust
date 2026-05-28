//! Regression test for <https://github.com/rust-lang/rust/issues/119382>.
//!
//! The `-Wrust-2021-incompatible-closure-captures` lint used to ICE
//! when applied to erroneous code.

//@ edition: 2018
//@ compile-flags: -Wrust-2021-incompatible-closure-captures

struct V(&mut i32);
//~^ ERROR missing lifetime specifier

fn nested(v: &V) {
    || {
        V(_somename) = v;
        //~^ ERROR cannot find value `_somename` in this scope
        //~| ERROR mismatched types
        v.0 = 0;
    };
}

fn main() {}
