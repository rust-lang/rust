// run-rustfix

#![allow(clippy::bind_instead_of_map)]

fn main() {
    let opt = Some(1);

    // Check `OPTION_MAP_OR_NONE`.
    // Single line case.
    let _ = opt.map_or(None, |x| Some(x + 1));
    // Multi-line case.
    #[rustfmt::skip]
    let _ = opt.map_or(None, |x| {
                        Some(x + 1)
                       });
}
