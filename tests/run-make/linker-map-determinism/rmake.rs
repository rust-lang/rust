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

    // Linker maps must be identical across repeated builds, including the
    // paths recorded for synthetic metadata and linked-symbol objects.
    assert_eq!(
        fs::read("first.map").unwrap(),
        fs::read("second.map").unwrap(),
        "complete linker maps differ between identical builds"
    );

    // The metadata object is linked for this dylib, while the symbols
    // object is only generated when linked symbols need to be retained.
    let metadata = "libfoo.rustc_link_metadata.rcgu.o";
    let symbols = "libfoo.rustc_link_symbols.rcgu.o";

    for map in ["first.map", "second.map"] {
        let contents = fs::read_to_string(map).unwrap();
        assert!(contents.contains(metadata), "metadata object missing from {map}");
    }

    assert_eq!(first_binary, second_binary, "linked dylibs differ between identical builds");

    // Normal linking removes synthetic objects.
    assert!(!fs::exists(metadata).unwrap());
    assert!(!fs::exists(symbols).unwrap());

    // -C save-temps retains them.
    rustc().input("lib.rs").crate_type("dylib").output("libfoo.so").arg("-Csave-temps").run();

    assert!(fs::exists(metadata).unwrap());

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
