//@ ignore-cross-compile
//@ only-linux
//@ ignore-backends: gcc

use run_make_support::{
    build_native_static_lib, cc, llvm_ar, llvm_nm, rfs, run, rust_lib_name, rustc, static_lib_name,
};

fn members(archive: &str) -> Vec<String> {
    llvm_ar().arg("t").arg(archive).run().stdout_utf8().lines().map(str::to_owned).collect()
}

fn count(members: &[String], name: &str) -> usize {
    members.iter().filter(|m| *m == name).count()
}

fn build_dup_archive(lib: &str, func: &str) {
    rfs::create_dir(lib);
    rfs::write(format!("{lib}/dup.c"), format!("int {func}(void) {{ return 0; }}\n"));
    cc().arg("-c").out_exe(&format!("{lib}/dup.o")).input(format!("{lib}/dup.c")).run();
    llvm_ar().obj_to_ar().output_input(static_lib_name(lib), format!("{lib}/dup.o")).run();
}

fn main() {
    build_native_static_lib("one");
    build_native_static_lib("two");
    build_native_static_lib("shared");
    build_dup_archive("dup_a", "dup_a_fn");
    build_dup_archive("dup_b", "dup_b_fn");

    rustc().input("dep.rs").run();
    let rlib = members(&rust_lib_name("dep"));
    assert_eq!(count(&rlib, "one.o"), 1, "{rlib:?}");
    assert_eq!(count(&rlib, "two.o"), 1, "{rlib:?}");
    assert_eq!(count(&rlib, "shared.o"), 1, "{rlib:?}");
    assert_eq!(count(&rlib, "dup.o"), 2, "{rlib:?}");
    for packed in ["one", "two", "shared", "dup_a", "dup_b"] {
        assert_eq!(count(&rlib, &static_lib_name(packed)), 0, "{rlib:?}");
    }

    for lto in [None, Some("thin"), Some("fat")] {
        for (cfg, expected) in [("use_one", "value=1"), ("use_two", "value=2")] {
            let mut cmd = rustc();
            cmd.input("main.rs").cfg(cfg);
            if let Some(lto) = lto {
                cmd.arg(format!("-Clto={lto}"));
            }
            cmd.run();
            run("main").assert_stdout_contains(expected);
        }
    }

    rustc().input("upstream_staticlib.rs").cfg("use_two").cfg("shared_y").cfg("dup_a").run();
    let staticlib = members(&static_lib_name("upstream_staticlib"));
    assert_eq!(count(&staticlib, "one.o"), 0, "{staticlib:?}");
    assert_eq!(count(&staticlib, "two.o"), 1, "{staticlib:?}");
    assert_eq!(count(&staticlib, "shared.o"), 1, "{staticlib:?}");
    assert_eq!(count(&staticlib, "dup.o"), 1, "{staticlib:?}");
    llvm_nm()
        .input(static_lib_name("upstream_staticlib"))
        .run()
        .assert_stdout_contains("dup_a_fn")
        .assert_stdout_not_contains("dup_b_fn");

    rustc().input("upstream_staticlib.rs").cfg("use_two").run();
    let staticlib = members(&static_lib_name("upstream_staticlib"));
    assert_eq!(count(&staticlib, "shared.o"), 0, "{staticlib:?}");
    assert_eq!(count(&staticlib, "dup.o"), 0, "{staticlib:?}");

    rustc().input("local_staticlib.rs").cfg("use_one").run();
    let staticlib = members(&static_lib_name("local_staticlib"));
    assert_eq!(count(&staticlib, "one.o"), 1, "{staticlib:?}");

    rustc().input("local_staticlib.rs").run();
    let staticlib = members(&static_lib_name("local_staticlib"));
    assert_eq!(count(&staticlib, "one.o"), 0, "{staticlib:?}");
}
