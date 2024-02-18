//@ compile-flags: --error-format human-annotate-rs -Z unstable-options
//@ error-pattern:cannot find type `Iter` in this scope

pub fn main() {
    let x: Iter;
}
