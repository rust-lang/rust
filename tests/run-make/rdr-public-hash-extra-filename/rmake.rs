// Changing `-C extra-filename` must change a crate's *public API hash*, even though the
// source is byte-for-byte identical between compilations.
//
// `extra_filename` is deliberately part of the crate's "global" public hash (see
// `HashableCrateRoot`): together with the stable crate id it is what distinguishes
// otherwise-identical builds of the same crate. Two versions of a crate can share a
// public API yet differ in private implementation, and a build system may be relying on
// `-C extra-filename` / `-C metadata` to keep them apart. Folding `extra_filename` into
// the public hash makes sure such crates never collide on it.
//
//@ ignore-cross-compile

use run_make_support::rustc;

/// Compile `dep.rs` with the public-api-hash machinery enabled and the given
/// `-C extra-filename` suffix, then return the public hash stored in the resulting
/// rlib's metadata header. The output rlib is named `libdep<suffix>.rlib`.
fn public_hash_with_extra_filename(suffix: &str) -> String {
    rustc().input("dep.rs").arg("-Zpublic-api-hash").extra_filename(suffix).run();

    let rlib = format!("libdep{suffix}.rlib");
    let listing = rustc().arg("-Zls=public_hash").input(&rlib).run().stdout_utf8();

    listing
        .lines()
        .find_map(|line| line.strip_prefix("Public hash:"))
        .map(|hash| hash.trim().to_owned())
        .expect("`-Zls=public_hash` did not print a `Public hash:` line")
}

fn main() {
    let baseline = public_hash_with_extra_filename("");
    let suffixed = public_hash_with_extra_filename("-suffix");
    let other = public_hash_with_extra_filename("-other");

    assert_ne!(
        baseline, suffixed,
        "public hash did not change when a `-C extra-filename` suffix was added",
    );
    assert_ne!(
        suffixed, other,
        "public hash did not change between two different `-C extra-filename` suffixes",
    );
}
