// Check that temporary object filenames do not affect the final rlib.
// In particular, archive members, lib.rmeta-link, and CodeView S_OBJNAME
// must not depend on the compiler invocation or output directory.

use run_make_support::{rfs, rustc};

fn main() {
    for debuginfo in ["0", "2"] {
        let first_dir = format!("debug-{debuginfo}-a");
        let second_dir = format!("debug-{debuginfo}-b");

        rfs::create_dir(&first_dir);
        rfs::create_dir(&second_dir);

        let first = format!("{first_dir}/librepro_rlib.rlib");
        let second = format!("{second_dir}/librepro_rlib.rlib");

        for output in [&first, &second] {
            rustc()
                .input("lib.rs")
                .crate_name("repro_rlib")
                .crate_type("rlib")
                .arg(format!("-Cdebuginfo={debuginfo}"))
                .arg("-Ccodegen-units=2")
                .arg("-Cincremental=off")
                .output(output)
                .run();
        }

        assert!(
            rfs::read(&first) == rfs::read(&second),
            "rlib differs across output directories with debuginfo={debuginfo}"
        );
    }
}
