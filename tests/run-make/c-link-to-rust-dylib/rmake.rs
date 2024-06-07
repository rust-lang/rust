// This test checks that C linking with Rust does not encounter any errors, with dynamic libraries.
// See <https://github.com/rust-lang/rust/issues/10434>.

//@ ignore-cross-compile

use std::fs::remove_file;

use run_make_support::{cc, cwd, dynamic_lib_extension, is_msvc, read_dir, run, run_fail, rustc};

fn main() {
    rustc().input("foo.rs").run();

    if is_msvc() {
        let lib = "foo.dll.lib";

        cc().input("bar.c").arg(lib).out_exe("bar").run();
    } else {
        cc().input("bar.c").arg("-lfoo").output("bar").library_search_path(cwd()).run();
    }

    run("bar");

    let expected_extension = dynamic_lib_extension();
    read_dir(cwd(), |path| {
        if path.is_file()
            && path.extension().is_some_and(|ext| ext == expected_extension)
            && path.file_name().and_then(|name| name.to_str()).is_some_and(|name| {
                name.ends_with(".so") || name.ends_with(".dll") || name.ends_with(".dylib")
            })
        {
            remove_file(path).unwrap();
        }
    });
    run_fail("bar");
}
