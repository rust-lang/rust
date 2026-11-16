//@ needs-target-std

use run_make_support::{diff, rfs, rustc};

fn emit(source: &str, threads: usize, opt_level: &str) -> String {
    rfs::write("input.rs", source);
    rustc()
        .input("input.rs")
        .crate_type("lib")
        .opt_level(opt_level)
        .arg(format!("-Zthreads={threads}"))
        .emit("mir")
        .output("output.mir")
        .run();
    rfs::read_to_string("output.mir")
}

fn main() {
    let source = rfs::read_to_string("main.rs");
    for opt_level in ["0", "2"] {
        let serial = emit(&source, 1, opt_level);
        assert!(serial.contains("\"alloc123 alloc1 a2\""));
        assert!(serial.contains("+0x4<imm>"));
        assert!(serial.contains("+0x8<imm>"));
        rfs::write(format!("one-thread-opt{opt_level}.mir"), &serial);
        for iteration in 0..3 {
            let parallel = emit(&source, 4, opt_level);
            rfs::write(format!("four-threads-opt{opt_level}-{iteration}.mir"), &parallel);
            diff()
                .expected_text("one thread", &serial)
                .actual_text("four threads", &parallel)
                .run();
        }

        // Equal bytes must not hide a change in which allocation is referenced.
        let changed_alias =
            source.replace("= [&FIRST, &SECOND, &FIRST]", "= [&FIRST, &FIRST, &FIRST]");
        assert_ne!(changed_alias, source);
        assert_ne!(serial, emit(&changed_alias, 1, opt_level));

        let changed_offset = source.replace("= [&DATA[1], &DATA[2]]", "= [&DATA[2], &DATA[2]]");
        assert_ne!(changed_offset, source);
        assert_ne!(serial, emit(&changed_offset, 1, opt_level));
    }
}
