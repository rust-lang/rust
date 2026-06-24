// Changing `-C metadata` must change a crate's *public API hash*, even though the
// source is byte-for-byte identical between compilations.
//
// `StableCrateId` is deliberately part of the crate's "global" public hash (see
// `HashableCrateRoot`): it is what distinguishes otherwise-identical builds of the same crate.
// Two versions of a crate can share a public API yet differ in private implementation, and a
// build system may be relying on `-C metadata` to keep them apart.
//@ ignore-cross-compile

use run_make_support::rustc;

/// Compile `dep.rs` with the public-api-hash machinery enabled and the given `-C metadata`,
/// then return the public hash stored in the resulting rlib's metadata header.
fn public_hash_with_metadata(metadata: Option<&str>) -> String {
    if let Some(metadata) = metadata {
        rustc().input("dep.rs").arg("-Zpublic-api-hash").metadata(metadata).run();
    } else {
        rustc().input("dep.rs").arg("-Zpublic-api-hash").run();
    }

    let listing = rustc().arg("-Zls=public_hash").input("libdep.rlib").run().stdout_utf8();

    listing
        .lines()
        .find_map(|line| line.strip_prefix("Public hash:"))
        .map(|hash| hash.trim().to_owned())
        .expect("`-Zls=public_hash` did not print a `Public hash:` line")
}

fn main() {
    let baseline = public_hash_with_metadata(None);
    let suffixed = public_hash_with_metadata(Some("A"));
    let other = public_hash_with_metadata(Some("B"));

    assert_ne!(baseline, suffixed, "public hash did not change when a `-C metadata` was added",);
    assert_ne!(
        suffixed, other,
        "public hash did not change between two different `-C metadata` options",
    );
}
