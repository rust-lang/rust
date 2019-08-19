// edition:2018
// aux-build:test-macros.rs
// aux-build:dollar-crate-external.rs

// Anonymize unstable non-dummy spans while still showing dummy spans `0..0`.
// normalize-stdout-test "bytes\([^0]\w*\.\.(\w+)\)" -> "bytes(LO..$1)"
// normalize-stdout-test "bytes\((\w+)\.\.[^0]\w*\)" -> "bytes($1..HI)"

#[macro_use]
extern crate test_macros;
extern crate dollar_crate_external;

type S = u8;

mod local {
    macro_rules! local {
        () => {
            print_bang! {
                struct M($crate::S);
            }

            #[print_attr]
            struct A($crate::S);

            #[derive(Print)]
            struct D($crate::S); //~ ERROR the name `D` is defined multiple times
        };
    }

    local!();
}

mod external {
    use crate::dollar_crate_external;

    dollar_crate_external::external!(); //~ ERROR the name `D` is defined multiple times
}

fn main() {}
