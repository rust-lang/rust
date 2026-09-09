// Unlike the incremental `rdr` tests (which can only toggle `cfg`-gated code
// inside a single, byte-identical source file), this test genuinely rewrites a
// source file on disk between compilations and checks how the crate's *public
// API hash* reacts:
//
//   * Appending a *private* item to the end of a file must NOT change the
//     public hash.
//   * Adding a *public* item MUST change the public hash
//
// The private code is appended to a *submodule* file (`sub.rs`), not to the
// crate root (`dep.rs`). The crate root's span covers the whole of `dep.rs`, so
// any byte appended there is hashed unconditionally; only a submodule file lets
// us isolate "private code added at the end of a file".
//
// The public hash is read back out of the compiled rlib's metadata header with
// `-Zls=public_hash`, which prints a `Public hash: <svh>` line.

//@ ignore-cross-compile

use run_make_support::{rfs, rustc};

/// Compile `dep.rs` with the public-api-hash machinery enabled and return the
/// public hash stored in the resulting rlib's metadata header.
fn compile_and_read_public_hash() -> String {
    rustc().input("dep.rs").arg("-Zpublic-api-hash").run();

    let listing = rustc().arg("-Zls=public_hash").input("libdep.rlib").run().stdout_utf8();

    listing
        .lines()
        .find_map(|line| line.strip_prefix("Public hash:"))
        .map(|hash| hash.trim().to_owned())
        .expect("`-Zls=public_hash` did not print a `Public hash:` line")
}

/// Append a line to the submodule source file `sub.rs`.
fn append_to_submodule(line: &str) {
    let mut contents = rfs::read_to_string("sub.rs");
    contents.push_str(line);
    contents.push('\n');
    rfs::write("sub.rs", contents);
}

fn main() {
    let baseline = compile_and_read_public_hash();

    append_to_submodule("fn private_at_end() {}");
    let after_private = compile_and_read_public_hash();
    assert_eq!(
        baseline, after_private,
        "public hash changed after appending a private item to the end of a file",
    );

    append_to_submodule("pub fn new_public() {}");
    let after_public = compile_and_read_public_hash();
    assert_ne!(
        after_private, after_public,
        "public hash did not change after adding a public item",
    );
}
