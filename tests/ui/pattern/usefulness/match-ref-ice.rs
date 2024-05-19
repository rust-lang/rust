#![deny(unreachable_patterns)]

// The arity of `ref x` is always 1. If the pattern is compared to some non-structural type whose
// arity is always 0, an ICE occurs.
//
// Related issue: #23009

fn main() {
    let homura = [1, 2, 3];

    match homura {
        [1, ref _madoka, 3] => (),
        [1, 2, 3] => (), //~ ERROR unreachable pattern
        [_, _, _] => (),
    }
}
