// run-rustfix

#![warn(clippy::match_like_matches_macro)]
#![allow(unreachable_patterns)]

fn main() {
    let x = Some(5);

    // Lint
    let _y = match x {
        Some(0) => true,
        _ => false,
    };

    // Lint
    let _w = match x {
        Some(_) => true,
        _ => false,
    };

    // Turn into is_none
    let _z = match x {
        Some(_) => false,
        None => true,
    };

    // Lint
    let _zz = match x {
        Some(r) if r == 0 => false,
        _ => true,
    };

    // Lint
    let _zzz = if let Some(5) = x { true } else { false };

    // No lint
    let _a = match x {
        Some(_) => false,
        _ => false,
    };

    // No lint
    let _ab = match x {
        Some(0) => false,
        _ => true,
        None => false,
    };
}
