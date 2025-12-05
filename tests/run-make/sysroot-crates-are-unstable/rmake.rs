// Check that crates in the sysroot are treated as unstable, unless they are
// on a list of known-stable sysroot crates.

use std::path::{Path, PathBuf};
use std::str;

use run_make_support::{rfs, rustc, target};

fn is_stable_crate(name: &str) -> bool {
    matches!(name, "std" | "alloc" | "core" | "proc_macro")
}

fn main() {
    for cr in get_unstable_sysroot_crates() {
        check_crate_is_unstable(&cr);
    }
    println!("Done");
}

#[derive(Debug)]
struct Crate {
    name: String,
    path: PathBuf,
}

fn check_crate_is_unstable(cr: &Crate) {
    let Crate { name, path } = cr;

    print!("- Verifying that sysroot crate '{name}' is an unstable crate ...");

    // Trying to use this crate from a user program should fail.
    let output = rustc()
        .crate_type("rlib")
        .extern_(name, path)
        .input("-")
        .stdin_buf(format!("extern crate {name};"))
        .run_fail();

    // Make sure it failed for the intended reason, not some other reason.
    // (The actual feature required varies between crates.)
    output.assert_stderr_contains("use of unstable library feature");

    println!(" OK");
}

fn get_unstable_sysroot_crates() -> Vec<Crate> {
    let sysroot = PathBuf::from(rustc().print("sysroot").run().stdout_utf8().trim());
    let sysroot_libs_dir = sysroot.join("lib").join("rustlib").join(target()).join("lib");
    println!("Sysroot libs dir: {sysroot_libs_dir:?}");

    // Generate a list of all library crates in the sysroot.
    let sysroot_crates = get_all_crates_in_dir(&sysroot_libs_dir);
    println!(
        "Found {} sysroot crates: {:?}",
        sysroot_crates.len(),
        sysroot_crates.iter().map(|cr| &cr.name).collect::<Vec<_>>()
    );

    // Self-check: If we didn't find `core`, we probably checked the wrong directory.
    assert!(
        sysroot_crates.iter().any(|cr| cr.name == "core"),
        "Couldn't find `core` in {sysroot_libs_dir:?}"
    );

    let unstable_sysroot_crates =
        sysroot_crates.into_iter().filter(|cr| !is_stable_crate(&cr.name)).collect::<Vec<_>>();
    // Self-check: There should be at least one unstable crate in the directory.
    assert!(
        !unstable_sysroot_crates.is_empty(),
        "Couldn't find any unstable crates in {sysroot_libs_dir:?}"
    );
    unstable_sysroot_crates
}

fn get_all_crates_in_dir(libs_dir: &Path) -> Vec<Crate> {
    let mut libs = vec![];
    rfs::read_dir_entries(libs_dir, |path| {
        if !path.is_file() {
            return;
        }
        if let Some(name) = crate_name_from_path(path) {
            libs.push(Crate { name, path: path.to_owned() });
        }
    });
    libs.sort_by(|a, b| a.name.cmp(&b.name));
    libs
}

/// Treat a file as a crate if its name begins with `lib` and ends with `.rlib`.
/// The crate name is the part before the first hyphen (if any).
fn crate_name_from_path(path: &Path) -> Option<String> {
    let name = path
        .file_name()?
        .to_str()?
        .strip_prefix("lib")?
        .strip_suffix(".rlib")?
        .split('-')
        .next()
        .expect("split always yields at least one string");
    Some(name.to_owned())
}
