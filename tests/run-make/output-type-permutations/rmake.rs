// In 2014, rustc's output flags were reworked to be a lot more modular.
// This test uses these output flags in an expansive variety of combinations
// and syntax styles, checking that compilation is successful and that output
// files are exactly what is expected, no more, no less.
// See https://github.com/rust-lang/rust/pull/12020

//@ ignore-cross-compile
// Reason: some cross-compiled targets don't support various crate types and fail to link.

use std::path::PathBuf;

use run_make_support::{
    bin_name, dynamic_lib_name, filename_not_in_denylist, rfs, rust_lib_name, rustc,
    shallow_find_files, static_lib_name,
};

// Each test takes 4 arguments:
// `must_exist`: output files which must be found - if any are absent, the test fails
// `can_exist`: optional output files which will not trigger a failure
// `dir`: the name of the directory where the test happens
// `rustc_invocation`: the rustc command being tested
// Any unexpected output files not listed in `must_exist` or `can_exist` will cause a failure.
#[track_caller]
fn assert_expected_output_files(expectations: Expectations, rustc_invocation: impl Fn()) {
    let Expectations { expected_files: must_exist, allowed_files: can_exist, test_dir: dir } =
        expectations;

    rfs::create_dir(&dir);
    rustc_invocation();
    for file in must_exist {
        rfs::remove_file(PathBuf::from(&dir).join(&file));
    }
    let actual_output_files =
        shallow_find_files(dir, |path| filename_not_in_denylist(path, &can_exist));
    if !&actual_output_files.is_empty() {
        dbg!(&actual_output_files);
        panic!("unexpected output artifacts detected");
    }
}

struct Expectations {
    /// Output files which must be found. The test fails if any are absent.
    expected_files: Vec<String>,
    /// Allowed output files which will not trigger a failure.
    allowed_files: Vec<String>,
    /// Name of the directory where the test happens.
    test_dir: String,
}

macro_rules! s {
    ( $( $x:expr ),* ) => {
        {
            let mut temp_vec = Vec::new();
            $(
                temp_vec.push($x.to_string());
            )*
            temp_vec
        }
    };
}

fn main() {
    let bin_foo = bin_name("foo");

    assert_expected_output_files(
        Expectations {
            expected_files: s![
                static_lib_name("bar"),
                dynamic_lib_name("bar"),
                rust_lib_name("bar")
            ],
            allowed_files: s![
                "libbar.dll.exp",
                "libbar.dll.lib",
                "libbar.pdb",
                "libbar.dll.a",
                "libbar.exe.a",
                "bar.dll.exp",
                "bar.dll.lib",
                "bar.pdb",
                "bar.dll.a",
                "bar.exe.a"
            ],
            test_dir: "three-crates".to_string(),
        },
        || {
            rustc()
                .input("foo.rs")
                .out_dir("three-crates")
                .crate_type("rlib,dylib,staticlib")
                .run();
        },
    );

    assert_expected_output_files(
        Expectations {
            expected_files: s![bin_name("bar")],
            allowed_files: s!["bar.pdb"],
            test_dir: "bin-crate".to_string(),
        },
        || {
            rustc().input("foo.rs").crate_type("bin").out_dir("bin-crate").run();
        },
    );

    assert_expected_output_files(
        Expectations {
            expected_files: s!["bar.ll", "bar.bc", "bar.s", "bar.o", bin_name("bar")],
            allowed_files: s!["bar.pdb"],
            test_dir: "all-emit".to_string(),
        },
        || {
            rustc().input("foo.rs").emit("asm,llvm-ir,llvm-bc,obj,link").out_dir("all-emit").run();
        },
    );

    assert_expected_output_files(
        Expectations {
            expected_files: s!["foo"],
            allowed_files: vec![],
            test_dir: "asm-emit".to_string(),
        },
        || {
            rustc().input("foo.rs").emit("asm").output("asm-emit/foo").run();
        },
    );
    assert_expected_output_files(
        Expectations {
            expected_files: s!["foo"],
            allowed_files: vec![],
            test_dir: "asm-emit2".to_string(),
        },
        || {
            rustc().input("foo.rs").emit("asm=asm-emit2/foo").run();
        },
    );
    assert_expected_output_files(
        Expectations {
            expected_files: s!["foo"],
            allowed_files: vec![],
            test_dir: "asm-emit3".to_string(),
        },
        || {
            rustc().input("foo.rs").arg("--emit=asm=asm-emit3/foo").run();
        },
    );

    assert_expected_output_files(
        Expectations {
            expected_files: s!["foo"],
            allowed_files: vec![],
            test_dir: "llvm-ir-emit".to_string(),
        },
        || {
            rustc().input("foo.rs").emit("llvm-ir").output("llvm-ir-emit/foo").run();
        },
    );
    assert_expected_output_files(
        Expectations {
            expected_files: s!["foo"],
            allowed_files: vec![],
            test_dir: "llvm-ir-emit2".to_string(),
        },
        || {
            rustc().input("foo.rs").emit("llvm-ir=llvm-ir-emit2/foo").run();
        },
    );
    assert_expected_output_files(
        Expectations {
            expected_files: s!["foo"],
            allowed_files: vec![],
            test_dir: "llvm-ir-emit3".to_string(),
        },
        || {
            rustc().input("foo.rs").arg("--emit=llvm-ir=llvm-ir-emit3/foo").run();
        },
    );

    assert_expected_output_files(
        Expectations {
            expected_files: s!["foo"],
            allowed_files: vec![],
            test_dir: "llvm-bc-emit".to_string(),
        },
        || {
            rustc().input("foo.rs").emit("llvm-bc").output("llvm-bc-emit/foo").run();
        },
    );
    assert_expected_output_files(
        Expectations {
            expected_files: s!["foo"],
            allowed_files: vec![],
            test_dir: "llvm-bc-emit2".to_string(),
        },
        || {
            rustc().input("foo.rs").emit("llvm-bc=llvm-bc-emit2/foo").run();
        },
    );
    assert_expected_output_files(
        Expectations {
            expected_files: s!["foo"],
            allowed_files: vec![],
            test_dir: "llvm-bc-emit3".to_string(),
        },
        || {
            rustc().input("foo.rs").arg("--emit=llvm-bc=llvm-bc-emit3/foo").run();
        },
    );

    assert_expected_output_files(
        Expectations {
            expected_files: s!["foo"],
            allowed_files: vec![],
            test_dir: "obj-emit".to_string(),
        },
        || {
            rustc().input("foo.rs").emit("obj").output("obj-emit/foo").run();
        },
    );
    assert_expected_output_files(
        Expectations {
            expected_files: s!["foo"],
            allowed_files: vec![],
            test_dir: "obj-emit2".to_string(),
        },
        || {
            rustc().input("foo.rs").emit("obj=obj-emit2/foo").run();
        },
    );
    assert_expected_output_files(
        Expectations {
            expected_files: s!["foo"],
            allowed_files: vec![],
            test_dir: "obj-emit3".to_string(),
        },
        || {
            rustc().input("foo.rs").arg("--emit=obj=obj-emit3/foo").run();
        },
    );

    assert_expected_output_files(
        Expectations {
            expected_files: s![&bin_foo],
            allowed_files: s!["foo.pdb"],
            test_dir: "link-emit".to_string(),
        },
        || {
            rustc().input("foo.rs").emit("link").output("link-emit/".to_owned() + &bin_foo).run();
        },
    );
    assert_expected_output_files(
        Expectations {
            expected_files: s![&bin_foo],
            allowed_files: s!["foo.pdb"],
            test_dir: "link-emit2".to_string(),
        },
        || {
            rustc().input("foo.rs").emit(&format!("link=link-emit2/{bin_foo}")).run();
        },
    );
    assert_expected_output_files(
        Expectations {
            expected_files: s![&bin_foo],
            allowed_files: s!["foo.pdb"],
            test_dir: "link-emit3".to_string(),
        },
        || {
            rustc().input("foo.rs").arg(&format!("--emit=link=link-emit3/{bin_foo}")).run();
        },
    );

    assert_expected_output_files(
        Expectations {
            expected_files: s!["foo"],
            allowed_files: vec![],
            test_dir: "rlib".to_string(),
        },
        || {
            rustc().crate_type("rlib").input("foo.rs").output("rlib/foo").run();
        },
    );
    assert_expected_output_files(
        Expectations {
            expected_files: s!["foo"],
            allowed_files: vec![],
            test_dir: "rlib2".to_string(),
        },
        || {
            rustc().crate_type("rlib").input("foo.rs").emit("link=rlib2/foo").run();
        },
    );
    assert_expected_output_files(
        Expectations {
            expected_files: s!["foo"],
            allowed_files: vec![],
            test_dir: "rlib3".to_string(),
        },
        || {
            rustc().crate_type("rlib").input("foo.rs").arg("--emit=link=rlib3/foo").run();
        },
    );

    assert_expected_output_files(
        Expectations {
            expected_files: s![bin_foo],
            allowed_files: s![
                "libfoo.dll.exp",
                "libfoo.dll.lib",
                "libfoo.pdb",
                "libfoo.dll.a",
                "libfoo.exe.a",
                "foo.dll.exp",
                "foo.dll.lib",
                "foo.pdb",
                "foo.dll.a",
                "foo.exe.a"
            ],
            test_dir: "dylib".to_string(),
        },
        || {
            rustc()
                .crate_type("dylib")
                .input("foo.rs")
                .output("dylib/".to_owned() + &bin_foo)
                .run();
        },
    );
    assert_expected_output_files(
        Expectations {
            expected_files: s![bin_foo],
            allowed_files: s![
                "libfoo.dll.exp",
                "libfoo.dll.lib",
                "libfoo.pdb",
                "libfoo.dll.a",
                "libfoo.exe.a",
                "foo.dll.exp",
                "foo.dll.lib",
                "foo.pdb",
                "foo.dll.a",
                "foo.exe.a"
            ],
            test_dir: "dylib2".to_string(),
        },
        || {
            rustc()
                .crate_type("dylib")
                .input("foo.rs")
                .emit(&format!("link=dylib2/{bin_foo}"))
                .run();
        },
    );
    assert_expected_output_files(
        Expectations {
            expected_files: s![bin_foo],
            allowed_files: s![
                "libfoo.dll.exp",
                "libfoo.dll.lib",
                "libfoo.pdb",
                "libfoo.dll.a",
                "libfoo.exe.a",
                "foo.dll.exp",
                "foo.dll.lib",
                "foo.pdb",
                "foo.dll.a",
                "foo.exe.a"
            ],
            test_dir: "dylib3".to_string(),
        },
        || {
            rustc()
                .crate_type("dylib")
                .input("foo.rs")
                .arg(&format!("--emit=link=dylib3/{bin_foo}"))
                .run();
        },
    );

    assert_expected_output_files(
        Expectations {
            expected_files: s!["foo"],
            allowed_files: vec![],
            test_dir: "staticlib".to_string(),
        },
        || {
            rustc().crate_type("staticlib").input("foo.rs").output("staticlib/foo").run();
        },
    );
    assert_expected_output_files(
        Expectations {
            expected_files: s!["foo"],
            allowed_files: vec![],
            test_dir: "staticlib2".to_string(),
        },
        || {
            rustc().crate_type("staticlib").input("foo.rs").emit("link=staticlib2/foo").run();
        },
    );
    assert_expected_output_files(
        Expectations {
            expected_files: s!["foo"],
            allowed_files: vec![],
            test_dir: "staticlib3".to_string(),
        },
        || {
            rustc().crate_type("staticlib").input("foo.rs").arg("--emit=link=staticlib3/foo").run();
        },
    );

    assert_expected_output_files(
        Expectations {
            expected_files: s![bin_foo],
            allowed_files: s!["foo.pdb"],
            test_dir: "bincrate".to_string(),
        },
        || {
            rustc()
                .crate_type("bin")
                .input("foo.rs")
                .output("bincrate/".to_owned() + &bin_foo)
                .run();
        },
    );
    assert_expected_output_files(
        Expectations {
            expected_files: s![bin_foo],
            allowed_files: s!["foo.pdb"],
            test_dir: "bincrate2".to_string(),
        },
        || {
            rustc()
                .crate_type("bin")
                .input("foo.rs")
                .emit(&format!("link=bincrate2/{bin_foo}"))
                .run();
        },
    );
    assert_expected_output_files(
        Expectations {
            expected_files: s![bin_foo],
            allowed_files: s!["foo.pdb"],
            test_dir: "bincrate3".to_string(),
        },
        || {
            rustc()
                .crate_type("bin")
                .input("foo.rs")
                .arg(&format!("--emit=link=bincrate3/{bin_foo}"))
                .run();
        },
    );

    assert_expected_output_files(
        Expectations {
            expected_files: s!["ir", rust_lib_name("bar")],
            allowed_files: vec![],
            test_dir: "rlib-ir".to_string(),
        },
        || {
            rustc()
                .input("foo.rs")
                .emit("llvm-ir=rlib-ir/ir")
                .emit("link")
                .crate_type("rlib")
                .out_dir("rlib-ir")
                .run();
        },
    );

    assert_expected_output_files(
        Expectations {
            expected_files: s!["ir", "asm", "bc", "obj", "link"],
            allowed_files: vec![],
            test_dir: "staticlib-all".to_string(),
        },
        || {
            rustc()
                .input("foo.rs")
                .emit("asm=staticlib-all/asm")
                .emit("llvm-ir=staticlib-all/ir")
                .emit("llvm-bc=staticlib-all/bc")
                .emit("obj=staticlib-all/obj")
                .emit("link=staticlib-all/link")
                .crate_type("staticlib")
                .run();
        },
    );
    assert_expected_output_files(
        Expectations {
            expected_files: s!["ir", "asm", "bc", "obj", "link"],
            allowed_files: vec![],
            test_dir: "staticlib-all2".to_string(),
        },
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
        Expectations {
            expected_files: s!["bar.ll", "bar.s", "bar.o", static_lib_name("bar")],
            allowed_files: s!["bar.bc"], // keep this one for the next test
            test_dir: "staticlib-all3".to_string(),
        },
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
    assert_expected_output_files(
        Expectations {
            expected_files: s!["bar.bc", rust_lib_name("bar"), "foo.bc"],
            allowed_files: vec![],
            test_dir: "rlib-emits".to_string(),
        },
        || {
            rfs::rename("staticlib-all3/bar.bc", "rlib-emits/foo.bc");
            rustc()
                .input("foo.rs")
                .emit("llvm-bc,link")
                .crate_type("rlib")
                .out_dir("rlib-emits")
                .run();
            assert_eq!(rfs::read("rlib-emits/foo.bc"), rfs::read("rlib-emits/bar.bc"));
        },
    );
}
