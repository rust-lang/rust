//! Checks that all unstable library crates in the sysroot are actually treated
//! as unstable.
#![deny(warnings)]

use run_make_support::{env_var, read_dir, rustc};
use std::path::{Path, PathBuf};
use std::str;

#[derive(Debug)]
struct Lib {
    name: String,
    path: PathBuf,
}

fn check_lib(lib: &Lib) -> Result<(), ()> {
    let Lib { name, path } = lib;

    println!("verifying that sysroot crate '{name}' is an unstable crate");

    let output = rustc()
        .input("-")
        .crate_type("rlib")
        .target(&env_var("TARGET"))
        .extern_(name, path)
        .stdin(format!("extern crate {name};"))
        .run_unchecked();

    if !output.status().success()
        && output.stderr_utf8().contains("use of unstable library feature")
    {
        return Ok(());
    }

    eprintln!();
    eprintln!("CRATE IS NOT UNSTABLE: `{name}` at {path:?}");
    eprintln!("output status: `{}`", output.status());
    eprintln!("=== STDOUT ===");
    eprint!("{}", output.stdout_utf8());
    eprintln!("==============");
    eprintln!("=== STDERR ===");
    eprint!("{}", output.stderr_utf8());
    eprintln!("==============");

    Err(())
}

fn get_all_libs(libs_dir: &Path) -> Vec<Lib> {
    let mut libs = vec![];
    read_dir(libs_dir, |file| {
        if !file.is_file() {
            return;
        };

        // Treat a file as a library if it begins with `lib` and ends with `.rlib`.
        // The library name is the part before the first hyphen (if any).
        // FIXME: Use a `try` block once they're stable.
        let Some(lib_name) = Some(file).and_then(|file| {
            file.file_name()?
                .to_str()?
                .strip_prefix("lib")?
                .strip_suffix(".rlib")?
                .split('-')
                .next()
        }) else {
            return;
        };

        libs.push(Lib { name: lib_name.to_owned(), path: file.to_owned() });
    });
    libs
}

fn is_stable_crate(name: &str) -> bool {
    matches!(name, "std" | "alloc" | "core" | "proc_macro")
}

fn main() {
    // Generate a list of all library crates in the sysroot.
    let sysroot_libs_dir = PathBuf::from(env_var("SYSROOT_BASE"))
        .join("lib/rustlib")
        .join(env_var("TARGET"))
        .join("lib");
    let sysroot_libs = get_all_libs(&sysroot_libs_dir);

    // Self-check: If we didn't find `core`, we probably checked the wrong directory.
    assert!(
        sysroot_libs.iter().any(|lib| lib.name == "core"),
        "couldn't find `core` in {sysroot_libs_dir:?}:\n{sysroot_libs:#?}"
    );

    let unstable_sysroot_libs =
        sysroot_libs.iter().filter(|lib| !is_stable_crate(&lib.name)).collect::<Vec<_>>();
    // Self-check: There should be at least one unstable lib in the directory.
    assert!(
        !unstable_sysroot_libs.is_empty(),
        "couldn't find any unstable libs in {sysroot_libs_dir:?}:\n{sysroot_libs:#?}"
    );

    // Check all of the crates before actually failing, so that we report all
    // errors instead of just the first one.
    let results = unstable_sysroot_libs.iter().map(|lib| check_lib(lib)).collect::<Vec<_>>();
    if results.iter().any(|r| r.is_err()) {
        std::process::exit(1);
    }
}
