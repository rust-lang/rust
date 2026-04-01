// Checks that the following does not ICE because `decorate` is incorrectly skipped due to
// `--force-warn`.

//@ compile-flags: -Dunused_must_use -Awarnings --force-warn unused_must_use --crate-type=lib
//@ check-pass

#[must_use]
fn f() {}

pub fn g() {
    f();
    //~^ WARN unused return value
}
