// In 2014, rustc's output flags were reworked to be a lot more modular.
// This test uses these output flags in an expansive variety of combinations
// and syntax styles, checking that compilation is successful and that output
// files are exactly what is expected, no more, no less.
// See https://github.com/rust-lang/rust/pull/12020

use run_make_support::{
    bin_name, cwd, dynamic_lib_name, fs_wrapper, name_not_among, rust_lib_name, rustc,
    shallow_find_files, static_lib_name,
};

// Each test takes 4 arguments:
// `must_exist`: output files which must be found - if any are absent, the test fails
// `can_exist`: optional output files which will not trigger a failure
// `dir`: the name of the directory where the test happens
// `rustc_invocation`: the rustc command being tested
// Any unexpected output files not listed in `must_exist` or `can_exist` will cause a failure.
fn assert_expected_output_files(
    must_exist: &[&'static str],
    can_exist: &[&'static str],
    dir: &str,
    rustc_invocation: impl Fn(),
) {
    fs_wrapper::create_dir(dir);
    rustc_invocation();
    for file in must_exist {
        fs_wrapper::remove_file(dir.to_owned() + "/" + file);
    }
    let actual_output_files = shallow_find_files(dir, |path| name_not_among(path, can_exist));
    if !&actual_output_files.is_empty() {
        dbg!(&actual_output_files);
        panic!("unexpected output artifacts detected");
    }
}

fn main() {
    let bin_foo = Box::leak(Box::new(bin_name("foo")));
    let bin_bar = Box::leak(Box::new(bin_name("bar")));
    let static_bar = Box::leak(Box::new(static_lib_name("bar")));
    let dynamic_bar = Box::leak(Box::new(dynamic_lib_name("bar")));
    let rust_bar = Box::leak(Box::new(rust_lib_name("bar")));

    assert_expected_output_files(
        &[static_bar, dynamic_bar, rust_bar],
        &[
            "libbar.ddl.exp",
            "libbar.dll.lib",
            "libbar.pdb",
            "libbar.dll.a",
            "libbar.exe.a",
            "bar.ddl.exp",
            "bar.dll.lib",
            "bar.pdb",
            "bar.dll.a",
            "bar.exe.a",
        ],
        "three-crates",
        || {
            rustc()
                .input("foo.rs")
                .out_dir("three-crates")
                .crate_type("rlib,dylib,staticlib")
                .run();
        },
    );

    assert_expected_output_files(&[bin_bar], &["bar.pdb"], "bin-crate", || {
        rustc().input("foo.rs").crate_type("bin").out_dir("bin-crate").run();
    });

    assert_expected_output_files(
        &["bar.ll", "bar.bc", "bar.s", "bar.o", bin_bar],
        &["bar.pdb"],
        "all-emit",
        || {
            rustc().input("foo.rs").emit("asm,llvm-ir,llvm-bc,obj,link").out_dir("all-emit").run();
        },
    );

    assert_expected_output_files(&["foo"], &[], "asm-emit", || {
        rustc().input("foo.rs").emit("asm").output("asm-emit/foo").run();
    });
    assert_expected_output_files(&["foo"], &[], "asm-emit2", || {
        rustc().input("foo.rs").emit("asm=asm-emit2/foo").run();
    });
    assert_expected_output_files(&["foo"], &[], "asm-emit3", || {
        rustc().input("foo.rs").arg("--emit=asm=asm-emit3/foo").run();
    });

    assert_expected_output_files(&["foo"], &[], "llvm-ir-emit", || {
        rustc().input("foo.rs").emit("llvm-ir").output("llvm-ir-emit/foo").run();
    });
    assert_expected_output_files(&["foo"], &[], "llvm-ir-emit2", || {
        rustc().input("foo.rs").emit("llvm-ir=llvm-ir-emit2/foo").run();
    });
    assert_expected_output_files(&["foo"], &[], "llvm-ir-emit3", || {
        rustc().input("foo.rs").arg("--emit=llvm-ir=llvm-ir-emit3/foo").run();
    });

    assert_expected_output_files(&["foo"], &[], "llvm-bc-emit", || {
        rustc().input("foo.rs").emit("llvm-bc").output("llvm-bc-emit/foo").run();
    });
    assert_expected_output_files(&["foo"], &[], "llvm-bc-emit2", || {
        rustc().input("foo.rs").emit("llvm-bc=llvm-bc-emit2/foo").run();
    });
    assert_expected_output_files(&["foo"], &[], "llvm-bc-emit3", || {
        rustc().input("foo.rs").arg("--emit=llvm-bc=llvm-bc-emit3/foo").run();
    });

    assert_expected_output_files(&["foo"], &[], "obj-emit", || {
        rustc().input("foo.rs").emit("obj").output("obj-emit/foo").run();
    });
    assert_expected_output_files(&["foo"], &[], "obj-emit2", || {
        rustc().input("foo.rs").emit("obj=obj-emit2/foo").run();
    });
    assert_expected_output_files(&["foo"], &[], "obj-emit3", || {
        rustc().input("foo.rs").arg("--emit=obj=obj-emit3/foo").run();
    });

    assert_expected_output_files(&[bin_foo], &[], "link-emit", || {
        rustc().input("foo.rs").emit("link").output("link-emit/".to_owned() + bin_foo).run();
    });
    assert_expected_output_files(&[bin_foo], &[], "link-emit2", || {
        rustc().input("foo.rs").emit(&format!("link=link-emit2/{bin_foo}")).run();
    });
    assert_expected_output_files(&[bin_foo], &[], "link-emit3", || {
        rustc().input("foo.rs").arg(&format!("--emit=link=link-emit3/{bin_foo}")).run();
    });

    assert_expected_output_files(&["foo"], &[], "rlib", || {
        rustc().crate_type("rlib").input("foo.rs").output("rlib/foo").run();
    });
    assert_expected_output_files(&["foo"], &[], "rlib2", || {
        rustc().crate_type("rlib").input("foo.rs").emit("link=rlib2/foo").run();
    });
    assert_expected_output_files(&["foo"], &[], "rlib3", || {
        rustc().crate_type("rlib").input("foo.rs").arg("--emit=link=rlib3/foo").run();
    });

    assert_expected_output_files(
        &[bin_foo],
        &[
            "libbar.ddl.exp",
            "libbar.dll.lib",
            "libbar.pdb",
            "libbar.dll.a",
            "libbar.exe.a",
            "bar.ddl.exp",
            "bar.dll.lib",
            "bar.pdb",
            "bar.dll.a",
            "bar.exe.a",
        ],
        "dylib",
        || {
            rustc().crate_type("dylib").input("foo.rs").output("dylib/".to_owned() + bin_foo).run();
        },
    );
    assert_expected_output_files(
        &[bin_foo],
        &[
            "libbar.ddl.exp",
            "libbar.dll.lib",
            "libbar.pdb",
            "libbar.dll.a",
            "libbar.exe.a",
            "bar.ddl.exp",
            "bar.dll.lib",
            "bar.pdb",
            "bar.dll.a",
            "bar.exe.a",
        ],
        "dylib2",
        || {
            rustc()
                .crate_type("dylib")
                .input("foo.rs")
                .emit(&format!("link=dylib2/{bin_foo}"))
                .run();
        },
    );
    assert_expected_output_files(
        &[bin_foo],
        &[
            "libbar.ddl.exp",
            "libbar.dll.lib",
            "libbar.pdb",
            "libbar.dll.a",
            "libbar.exe.a",
            "bar.ddl.exp",
            "bar.dll.lib",
            "bar.pdb",
            "bar.dll.a",
            "bar.exe.a",
        ],
        "dylib3",
        || {
            rustc()
                .crate_type("dylib")
                .input("foo.rs")
                .arg(&format!("--emit=link=dylib3/{bin_foo}"))
                .run();
        },
    );

    assert_expected_output_files(&["foo"], &[], "staticlib", || {
        rustc().crate_type("staticlib").input("foo.rs").output("staticlib/foo").run();
    });
    assert_expected_output_files(&["foo"], &[], "staticlib2", || {
        rustc().crate_type("staticlib").input("foo.rs").emit("link=staticlib2/foo").run();
    });
    assert_expected_output_files(&["foo"], &[], "staticlib3", || {
        rustc().crate_type("staticlib").input("foo.rs").arg("--emit=link=staticlib3/foo").run();
    });

    assert_expected_output_files(&["foo"], &["foo.pdb"], "bincrate", || {
        rustc().crate_type("bin").input("foo.rs").output("bincrate/".to_owned() + bin_foo).run();
    });
    assert_expected_output_files(&["foo"], &["foo.pdb"], "bincrate2", || {
        rustc().crate_type("bin").input("foo.rs").emit(&format!("link=bincrate2/{bin_foo}")).run();
    });
    assert_expected_output_files(&["foo"], &["foo.pdb"], "bincrate3", || {
        rustc()
            .crate_type("bin")
            .input("foo.rs")
            .arg(&format!("--emit=link=bincrate3/{bin_foo}"))
            .run();
    });

    assert_expected_output_files(&["ir", rust_bar], &[], "rlib-ir", || {
        rustc()
            .input("foo.rs")
            .emit("llvm-ir=rlib-ir/ir")
            .emit("link")
            .crate_type("rlib")
            .out_dir("rlib-ir")
            .run();
    });

    assert_expected_output_files(&["ir", "asm", "bc", "obj", "link"], &[], "staticlib-all", || {
        rustc()
            .input("foo.rs")
            .emit("asm=staticlib-all/asm")
            .emit("llvm-ir=staticlib-all/ir")
            .emit("llvm-bc=staticlib-all/bc")
            .emit("obj=staticlib-all/obj")
            .emit("link=staticlib-all/link")
            .crate_type("staticlib")
            .run();
    });
    assert_expected_output_files(
        &["ir", "asm", "bc", "obj", "link"],
        &[],
        "staticlib-all2",
        || {
            rustc()
                .input("foo.rs")
                .arg("--emit=asm=staticlib-all2/asm")
                .arg("--emit")
                .arg("llvm-ir=staticlib-all2/ir")
                .arg("--emit=llvm-bc=staticlib-all2/bc")
                .arg("--emit")
                .arg("obj=staticlib-all2/obj")
                .arg("--emit=link=staticlib-all2/link")
                .crate_type("staticlib")
                .run();
        },
    );

    assert_expected_output_files(
        &["bar.ll", "bar.s", "bar.o", static_bar],
        &["bar.bc"], // keep this one for the next test
        "staticlib-all3",
        || {
            rustc()
                .input("foo.rs")
                .emit("asm,llvm-ir,llvm-bc,obj,link")
                .crate_type("staticlib")
                .out_dir("staticlib-all3")
                .run();
        },
    );

    // the .bc file from the previous test should be equivalent to this one, despite the difference
    // in crate type
    assert_expected_output_files(&["bar.bc", rust_bar, "foo.bc"], &[], "rlib-emits", || {
        fs_wrapper::rename("staticlib-all3/bar.bc", "rlib-emits/foo.bc");
        rustc().input("foo.rs").emit("llvm-bc,link").crate_type("rlib").out_dir("rlib-emits").run();
        assert_eq!(fs_wrapper::read("rlib-emits/foo.bc"), fs_wrapper::read("rlib-emits/bar.bc"));
    });
}
