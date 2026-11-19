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

    // Both synthetic linker inputs must use regular object filenames.
    let metadata = "libfoo.rustc_link_metadata.rcgu.o";
    let symbols = "libfoo.rustc_link_symbols.rcgu.o";

    for map in ["first.map", "second.map"] {
        let contents = fs::read_to_string(map).unwrap();
        assert!(contents.contains(metadata), "metadata object missing from {map}");
        assert!(contents.contains(symbols), "symbols object missing from {map}");
    }

    assert_eq!(first_binary, second_binary, "linked dylibs differ between identical builds");

    // Normal linking removes synthetic objects.
    assert!(!fs::exists(metadata).unwrap());
    assert!(!fs::exists(symbols).unwrap());

    // -C save-temps retains them.
    rustc().input("lib.rs").crate_type("dylib").output("libfoo.so").arg("-Csave-temps").run();

    assert!(fs::exists(metadata).unwrap());
    assert!(fs::exists(symbols).unwrap());

    // A subsequent normal build removes them again.
    rustc().input("lib.rs").crate_type("dylib").output("libfoo.so").run();

    assert!(!fs::exists(metadata).unwrap());
    assert!(!fs::exists(symbols).unwrap());

    // A linker failure must also clean up the generated objects.
    rustc()
        .input("lib.rs")
        .crate_type("dylib")
        .output("libfoo.so")
        .arg("-Clink-arg=-Wl,--invalid-linker-map-test-option")
        .run_fail();

    assert!(!fs::exists(metadata).unwrap());
    assert!(!fs::exists(symbols).unwrap());
}
