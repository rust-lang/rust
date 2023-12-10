#![feature(never_type)]
#![feature(exhaustive_patterns)]
#![deny(unreachable_patterns)]

fn main() {}

fn foo(nevers: &[!]) {
    match nevers {
        &[] => (),
    };

    match nevers {
        &[] => (),
        &[_] => (),
        &[_, _, ..] => (),
    };

    match nevers {
        //~^ ERROR non-exhaustive patterns: `&[]` not covered
        &[_] => (),
    };
}
