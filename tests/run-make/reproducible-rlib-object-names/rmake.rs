// Check that temporary object filenames and -C extra-filename do not affect
// the final rlib or LTO link order.
// In particular:
// 1. Archive members, lib.rmeta-link, and CodeView S_OBJNAME must not depend
//    on the compiler invocation or output directory.
// 2. Swapping -C extra-filename across duplicate-named dependencies must not
//    change module sorting during LTO (#165736).

use run_make_support::{rfs, rustc};

fn main() {
    // Phase 1: Verify rlib byte-for-byte reproducibility across directories
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

    // Phase 2: Verify LTO link order determinism with duplicate crate names
    // and swapped extra-filename (#165736)
    for lto in ["thin", "fat"] {
        // Run 1: dep_a gets -1111, dep_b gets -2222
        rustc()
            .input("dep_a.rs")
            .crate_name("dep")
            .crate_type("rlib")
            .arg("-Cmetadata=1")
            .arg("-Cextra-filename=-1111")
            .out_dir(".")
            .run();
        rustc()
            .input("dep_b.rs")
            .crate_name("dep")
            .crate_type("rlib")
            .arg("-Cmetadata=2")
            .arg("-Cextra-filename=-2222")
            .out_dir(".")
            .run();
        rustc()
            .input("main.rs")
            .crate_type("bin")
            .arg(format!("-Clto={lto}"))
            .extern_("dep_one", "libdep-1111.rlib")
            .extern_("dep_two", "libdep-2222.rlib")
            .output(format!("bin_run1_{lto}"))
            .run();

        // Run 2: Swap extra-filenames (dep_a gets -2222, dep_b gets -1111)
        rustc()
            .input("dep_a.rs")
            .crate_name("dep")
            .crate_type("rlib")
            .arg("-Cmetadata=1")
            .arg("-Cextra-filename=-2222")
            .out_dir(".")
            .run();
        rustc()
            .input("dep_b.rs")
            .crate_name("dep")
            .crate_type("rlib")
            .arg("-Cmetadata=2")
            .arg("-Cextra-filename=-1111")
            .out_dir(".")
            .run();
        rustc()
            .input("main.rs")
            .crate_type("bin")
            .arg(format!("-Clto={lto}"))
            .extern_("dep_one", "libdep-2222.rlib")
            .extern_("dep_two", "libdep-1111.rlib")
            .output(format!("bin_run2_{lto}"))
            .run();

        assert!(
            rfs::read(format!("bin_run1_{lto}")) == rfs::read(format!("bin_run2_{lto}")),
            "LTO binary differs when extra-filename is swapped across duplicate deps with lto={lto}"
        );
    }
}
