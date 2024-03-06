//@run-pass
#![deny(rust_2021_incompatible_closure_captures)]
#![allow(unused_must_use)]

fn filter_try_fold(
    predicate: &mut impl FnMut() -> bool,
) -> impl FnMut() -> bool + '_ {
    move || predicate()
}

fn main() {
    filter_try_fold(&mut || true);
}
