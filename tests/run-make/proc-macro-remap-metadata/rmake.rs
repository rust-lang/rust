//@ only-linux
//@ ignore-cross-compile
//@ needs-crate-type: proc-macro
//@ needs-dynamic-linking

// Regression test for #120825, fixed by #121959.
// Proc-macro metadata must respect --remap-path-prefix rather than
// recording the physical source checkout path.

use std::fs;
use std::path::{Path, PathBuf};

use run_make_support::object::read::{Object, ObjectSection};
use run_make_support::{cwd, dynamic_lib_name, rustc};

fn main() {
    let root = cwd();

    let first = build(&root.join("first"));
    let second = build(&root.join("second"));

    assert!(!first.is_empty(), "empty .rustc metadata section");
    assert_eq!(first, second, "proc-macro .rustc metadata depends on the physical source path");
}

fn build(dir: &Path) -> Vec<u8> {
    fs::create_dir_all(dir).unwrap();

    let source = dir.join("lib.rs");
    fs::write(
        &source,
        r#"
        extern crate proc_macro;

        use proc_macro::TokenStream;

        #[proc_macro]
        pub fn identity(input: TokenStream) -> TokenStream {
            helper();
            input
        }

        fn helper() {}
        "#,
    )
    .unwrap();

    let output: PathBuf = dir.join(dynamic_lib_name("repro_proc_macro"));

    rustc()
        .input(&source)
        .crate_name("repro_proc_macro")
        .crate_type("proc-macro")
        .edition("2021")
        .arg("-Cdebuginfo=2")
        .remap_path_prefix(dir, "/rustc/repro")
        .output(&output)
        .run();

    let bytes = fs::read(&output).unwrap();
    let object = run_make_support::object::File::parse(&*bytes).unwrap();

    let section = object.section_by_name(".rustc").expect("proc-macro dylib has no .rustc section");

    let metadata = section.data().unwrap().to_vec();

    // The absolute build directory must not survive in crate metadata.
    let path = dir.to_string_lossy();
    assert!(
        !metadata.windows(path.len()).any(|window| window == path.as_bytes()),
        "unremapped physical source path in .rustc metadata: {path}"
    );

    // Verify that the virtual source path is actually stored in metadata.
    let remapped = b"/rustc/repro/lib.rs";
    assert!(
        metadata.windows(remapped.len()).any(|window| window == remapped),
        "remapped source path missing from proc-macro metadata"
    );

    metadata
}
