//run-pass
#![deny(disjoint_capture_migration)]
#![allow(unused_must_use)]

fn filter_try_fold(
    predicate: &mut impl FnMut() -> bool,
) -> impl FnMut() -> bool + '_ {
    move || predicate()
}

fn main() {
    filter_try_fold(&mut || true);
}
