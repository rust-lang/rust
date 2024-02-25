//@ revisions: min_exhaustive_patterns exhaustive_patterns
#![cfg_attr(exhaustive_patterns, feature(exhaustive_patterns))]
#![cfg_attr(min_exhaustive_patterns, feature(min_exhaustive_patterns))]
#![feature(never_type)]
#![deny(unreachable_patterns)]

fn main() {}

fn foo(nevers: &[!]) {
    match nevers {
        //[min_exhaustive_patterns]~^ ERROR non-exhaustive patterns: `&[_, ..]` not covered
        &[] => (),
    };

    match nevers {
        &[] => (),
        &[_] => (),
        &[_, _, ..] => (),
    };

    match nevers {
        //[exhaustive_patterns]~^ ERROR non-exhaustive patterns: `&[]` not covered
        //[min_exhaustive_patterns]~^^ ERROR non-exhaustive patterns: `&[]` and `&[_, _, ..]` not covered
        &[_] => (),
    };
}
