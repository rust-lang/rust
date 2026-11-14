// Repeated dylib links must produce identical linker maps.
//
// Regression test for #142989: rustc used to pass a metadata object
// from a randomly named temporary directory to the linker, leaking
// that directory name into the map file.
//
//@ only-linux
//@ ignore-cross-compile
//@ ignore-backends: gcc

use std::fs;

use run_make_support::rustc;

fn main() {
    let build = |map: &str| -> Vec<u8> {
        let map_arg = format!("-Wl,-Map={map}");

        rustc().input("lib.rs").crate_type("dylib").output("libfoo.so").link_arg(&map_arg).run();

        fs::read("libfoo.so").unwrap()
    };

    let first_binary = build("first.map");
    let second_binary = build("second.map");

    // GNU ld records paths to both metadata and linked-symbol objects.
    // Neither pathname should vary between otherwise identical builds.
    assert!(
        fs::read("first.map").unwrap() == fs::read("second.map").unwrap(),
        "complete linker maps differ between identical builds"
    );

    let metadata_input = |map: &str| -> String {
        let map = fs::read_to_string(map).unwrap();
        map.lines()
            .find(|line| line.trim_start().starts_with(".rustc ") && line.contains(".o"))
            .expect("metadata object missing from linker map")
            .to_owned()
    };

    assert_eq!(
        metadata_input("first.map"),
        metadata_input("second.map"),
        "metadata object path differs between identical builds"
    );

    // Also verify that the linked-symbol object uses its stable path.
    for map in ["first.map", "second.map"] {
        let contents = fs::read_to_string(map).unwrap();
        assert!(
            contents.contains("libfoo.so.rustc-link-objects/symbols.o"),
            "stable symbols object path missing from {map}"
        );
        assert!(
            contents.contains("libfoo.so.rustc-link-objects/metadata.o"),
            "stable metadata object path missing from {map}"
        );
    }

    assert_eq!(first_binary, second_binary, "linked dylibs differ between identical builds");

    // Normal linking must remove compiler-generated temporary objects.
    let metadata = "libfoo.so.rustc-link-objects/metadata.o";
    let symbols = "libfoo.so.rustc-link-objects/symbols.o";
    let lock = "libfoo.so.rustc-link-objects.lock";

    assert!(!fs::exists(metadata).unwrap());
    assert!(!fs::exists(symbols).unwrap());
    assert!(fs::exists(lock).unwrap());

    // With save-temps, the metadata object must be retained.
    rustc().input("lib.rs").crate_type("dylib").output("libfoo.so").arg("-Csave-temps").run();

    assert!(fs::exists(metadata).unwrap());

    // The next ordinary build must replace and clean up stale objects.
    rustc().input("lib.rs").crate_type("dylib").output("libfoo.so").run();

    assert!(!fs::exists(metadata).unwrap());
    assert!(!fs::exists(symbols).unwrap());
    assert!(fs::exists(lock).unwrap());

    // An unrelated directory must not be adopted or modified.
    let unrelated = "libcollision.so.rustc-link-objects";
    fs::create_dir(unrelated).unwrap();
    fs::write(format!("{unrelated}/sentinel"), b"keep me").unwrap();

    rustc().input("lib.rs").crate_type("dylib").output("libcollision.so").run_fail();

    assert_eq!(fs::read(format!("{unrelated}/sentinel")).unwrap(), b"keep me");

    // Previously problematic sibling filenames must be left untouched.
    let sibling = "libfoo.so.rustc-rmeta.o";
    fs::write(sibling, b"unrelated file").unwrap();

    rustc().input("lib.rs").crate_type("dylib").output("libfoo.so").run();

    assert_eq!(fs::read(sibling).unwrap(), b"unrelated file");
}
