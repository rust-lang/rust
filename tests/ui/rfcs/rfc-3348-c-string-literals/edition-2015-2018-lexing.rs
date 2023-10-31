// Regression test for issue #113235.

// check-pass
// revisions: edition2015 edition2018
//[edition2015] edition: 2015
//[edition2018] edition: 2018

// Make sure that in pre-2021 editions we continue to parse the snippet
// `c"hello"` as an identifier followed by a (normal) string literal and
// allow the code below to compile.
// Prefixes including `c` as used by C string literals are only reserved
// in edition 2021 and onward.
//
// Consider checking out rust-2021/reserved-prefixes-migration.rs as well.

macro_rules! parse {
    (c $e:expr) => {
        $e
    };
}

fn main() {
    let _: &'static str = parse!(c"hello");
}
