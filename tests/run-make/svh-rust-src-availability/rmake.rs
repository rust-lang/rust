// Regression test for rust-lang/rust#153898.
//
// Installing rust-src must not change the crate's strict version hash or
// encoded metadata. In particular, local source availability can change
// whether formatting desugarings have precise source snippets.
//
//@ ignore-cross-compile
//@ needs-symlink

use std::path::Path;

use run_make_support::rfs::read_dir_entries;
use run_make_support::{bare_rustc, path, rfs, source_root};

fn shallow_symlink_dir(src: &Path, dst: &Path) {
    read_dir_entries(src, |entry| {
        let metadata = rfs::symlink_metadata(entry);
        let target = dst.join(entry.file_name().unwrap());

        if metadata.is_dir() {
            rfs::symlink_dir(entry, target);
        } else if metadata.is_file() {
            rfs::symlink_file(entry, target);
        } else if metadata.is_symlink() {
            rfs::copy_symlink(entry, target);
        }
    });
}

fn recreate_dir(dir: &Path) {
    if dir.exists() || dir.is_symlink() {
        rfs::recursive_remove(dir);
    }
    rfs::create_dir(dir);
}

fn compile(sysroot: &Path, output: &str) {
    bare_rustc()
        .sysroot(sysroot)
        .input("foo.rs")
        .crate_type("lib")
        .arg("--edition=2024")
        .arg("--emit=metadata")
        .arg("-Zmetadata-crate-hash=yes")
        .arg("-Zsimulate-remapped-rust-src-base=/rustc/xyz")
        .arg("-Ztranslate-remapped-path-to-local-path=yes")
        .arg("-o")
        .arg(output)
        .run();
}

fn main() {
    let real_sysroot = bare_rustc().print("sysroot").run().stdout_utf8();
    let real_sysroot = path(real_sysroot.trim());

    let fake = path("fakeroot");
    rfs::create_dir(&fake);

    // Reuse all real sysroot artifacts through symlinks, but give the fake
    // sysroot its own lib/rustlib/src directory.
    shallow_symlink_dir(&real_sysroot, &fake);

    let fake_lib = fake.join("lib");
    recreate_dir(&fake_lib);
    shallow_symlink_dir(&real_sysroot.join("lib"), &fake_lib);

    let fake_rustlib = fake_lib.join("rustlib");
    recreate_dir(&fake_rustlib);
    shallow_symlink_dir(&real_sysroot.join("lib/rustlib"), &fake_rustlib);

    let fake_src = fake_rustlib.join("src");
    recreate_dir(&fake_src);

    // First build: rust-src is unavailable.
    compile(&fake, "without.rmeta");

    // Second build: provide the checkout's standard library sources.
    let rust_src = fake_src.join("rust");
    rfs::create_dir(&rust_src);
    rfs::symlink_dir(source_root().join("library"), rust_src.join("library"));

    compile(&fake, "with.rmeta");

    let without = rfs::read("without.rmeta");
    let with = rfs::read("with.rmeta");

    assert_eq!(without, with, "crate metadata changed when rust-src became available");
}
