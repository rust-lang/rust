// When setting the crate type as a "bin" (in app.rs),
// this could cause a bug where some symbols would not be
// emitted in the object files. This has been fixed, and
// this test checks that the correct symbols have been successfully
// emitted inside the object files.
// See https://github.com/rust-lang/rust/issues/51671

use run_make_support::{nm, rustc, tmp_dir};

fn main() {
    rustc().emit("obj").input("app.rs").run();
    //FIXME(Oneirical): This should eventually be rmake_out_path
    let nm = nm(tmp_dir().join("app.o"));
    assert!(
        nm.contains("rust_begin_unwind")
            && nm.contains("rust_eh_personality")
            && nm.contains("__rg_oom")
    );
}
