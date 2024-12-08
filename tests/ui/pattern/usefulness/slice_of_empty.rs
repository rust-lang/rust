//@ revisions: normal exhaustive_patterns
#![cfg_attr(exhaustive_patterns, feature(exhaustive_patterns))]
#![feature(never_type)]
#![deny(unreachable_patterns)]

fn main() {}

fn foo(nevers: &[!]) {
    match nevers {
        //[normal]~^ ERROR non-exhaustive patterns: `&[_, ..]` not covered
        &[] => (),
    };

    match nevers {
        &[] => (),
        &[_] => (),
        &[_, _, ..] => (),
    };

    match nevers {
        //[exhaustive_patterns]~^ ERROR non-exhaustive patterns: `&[]` not covered
        //[normal]~^^ ERROR non-exhaustive patterns: `&[]` and `&[_, _, ..]` not covered
        &[_] => (),
    };
}
