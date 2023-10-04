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
        &[_] => (),        //~ ERROR unreachable pattern
        &[_, _, ..] => (), //~ ERROR unreachable pattern
    };

    match nevers {
        //~^ ERROR non-exhaustive patterns: `&[]` not covered
        &[_] => (), //~ ERROR unreachable pattern
    };
}
