use run_make_support::rfs::{create_dir, remove_file, rename};
use run_make_support::{Rustc, dynamic_lib_name, run_in_tmpdir, rust_lib_name, rustc};

#[derive(Debug, Copy, Clone)]
enum LibraryKind {
    Rlib,
    Dylib,
}

impl LibraryKind {
    fn crate_type(&self) -> &str {
        match self {
            LibraryKind::Rlib => "rlib",
            LibraryKind::Dylib => "dylib",
        }
    }

    fn add_extern(&self, rustc: &mut Rustc, dep_name: &str, dep_path: &str) {
        let dep_path = match self {
            LibraryKind::Dylib => format!("{dep_path}/{}", dynamic_lib_name(dep_name)),
            LibraryKind::Rlib => format!("{dep_path}/{}", rust_lib_name(dep_name)),
        };
        rustc.extern_(dep_name, dep_path);
    }
}

fn main() {
    // The compiler takes different paths based on if --extern is passed or not, so we test all
    // combinations (`rlib`/`dylib` x `--extern`/`no --extern`).
    for kind in [LibraryKind::Rlib, LibraryKind::Dylib] {
        eprintln!("Testing library kind {kind:?}");
        lookup_rmeta_in_lib_dir(kind);
        lookup_rmeta_in_lib_search_path(kind);
        lookup_rmeta_in_lib_dir_extern(kind);
        lookup_rmeta_in_lib_search_path_extern(kind);
    }
}

// Lookup .rmeta file in the same directory as a rlib/dylib with stub metadata.
fn lookup_rmeta_in_lib_dir(kind: LibraryKind) {
    run_in_tmpdir(|| {
        build_dep_rustc(kind).run();
        rustc().input("foo.rs").run();
    });
}

// Lookup .rmeta file in a directory passed to -L.
fn lookup_rmeta_in_lib_search_path(kind: LibraryKind) {
    run_in_tmpdir(|| {
        build_dep_rustc(kind).run();

        // Move the rmeta directory to a different path
        create_dir("rmeta-dir");
        rename("libdep1.rmeta", "rmeta-dir/libdep1.rmeta");

        rustc().input("foo.rs").library_search_path("rmeta-dir").run();
    });
}

// Lookup .rmeta file in the same directory as a rlib/dylib with stub metadata when specifying the
// dependency using --extern.
fn lookup_rmeta_in_lib_dir_extern(kind: LibraryKind) {
    run_in_tmpdir(|| {
        // Generate libdep1.rlib and libdep1.rmeta in deps
        create_dir("deps");
        build_dep_rustc(kind).out_dir("deps").run();

        let mut rustc = rustc();
        kind.add_extern(&mut rustc, "dep1", "deps");
        rustc.input("foo.rs").run();
    });
}

// Lookup .rmeta file in a directory passed to -L when specifying the dependency using --extern.
fn lookup_rmeta_in_lib_search_path_extern(kind: LibraryKind) {
    run_in_tmpdir(|| {
        create_dir("deps");
        build_dep_rustc(kind).out_dir("deps").run();

        // Move the rmeta directory to a different dir
        create_dir("rmeta-dir");
        rename("deps/libdep1.rmeta", "rmeta-dir/libdep1.rmeta");

        let mut rustc = rustc();
        kind.add_extern(&mut rustc, "dep1", "deps");
        rustc.library_search_path("rmeta-dir");
        rustc.input("foo.rs").run();
    });
}

fn build_dep_rustc(kind: LibraryKind) -> Rustc {
    let mut dep_rustc = rustc();
    dep_rustc
        .arg("-Zsplit-metadata")
        .crate_type(kind.crate_type())
        .input("dep1.rs")
        .emit("metadata,link");
    if matches!(kind, LibraryKind::Dylib) {
        dep_rustc.arg("-Cprefer-dynamic");
    }
    dep_rustc
}
