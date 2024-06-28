// In 2014, rustc's output flags were reworked to be a lot more modular.
// This test uses these output flags in an expansive variety of combinations
// and syntax styles, checking that compilation is successful and that no unexpected
// files are created.
// The assert_eq! checks that "1 file remains" at the end of each part of the test,
// because foo.rs counts as a file, and should be the only remaining one.
// See https://github.com/rust-lang/rust/pull/12020

use run_make_support::{
    bin_name, cwd, dynamic_lib_name, fs_wrapper, rust_lib_name, rustc, static_lib_name,
};

fn remove_artifacts() {
    std::fs::remove_file("libbar.ddl.exp").unwrap_or_default();
    std::fs::remove_file("libbar.dll.lib").unwrap_or_default();
    std::fs::remove_file("libbar.pdb").unwrap_or_default();
    std::fs::remove_file("libbar.dll.a").unwrap_or_default();
    std::fs::remove_file("libbar.exe.a").unwrap_or_default();
    std::fs::remove_file("bar.ddl.exp").unwrap_or_default();
    std::fs::remove_file("bar.dll.lib").unwrap_or_default();
    std::fs::remove_file("bar.pdb").unwrap_or_default();
    std::fs::remove_file("bar.dll.a").unwrap_or_default();
    std::fs::remove_file("bar.exe.a").unwrap_or_default();
}

fn main() {
    rustc().input("foo.rs").crate_type("rlib,dylib,staticlib").run();
    fs_wrapper::remove_file(rust_lib_name("bar"));
    fs_wrapper::remove_file(dynamic_lib_name("bar"));
    fs_wrapper::remove_file(static_lib_name("bar"));
    remove_artifacts();
    assert_eq!(fs_wrapper::read_dir(cwd()).count(), 1);

    rustc().input("foo.rs").crate_type("bin").run();
    fs_wrapper::remove_file(bin_name("bar"));
    std::fs::remove_file("bar.pdb").unwrap_or_default();
    assert_eq!(fs_wrapper::read_dir(cwd()).count(), 1);

    rustc().input("foo.rs").emit("asm,llvm-ir,llvm-bc,obj,link").run();
    fs_wrapper::remove_file("bar.ll");
    fs_wrapper::remove_file("bar.bc");
    fs_wrapper::remove_file("bar.s");
    fs_wrapper::remove_file("bar.o");
    fs_wrapper::remove_file(bin_name("bar"));
    std::fs::remove_file("bar.pdb").unwrap_or_default();
    assert_eq!(fs_wrapper::read_dir(cwd()).count(), 1);

    rustc().input("foo.rs").emit("asm").output("foo").run();
    fs_wrapper::remove_file("foo");
    rustc().input("foo.rs").emit("asm=foo").run();
    fs_wrapper::remove_file("foo");
    rustc().input("foo.rs").arg("--emit=asm=foo").run();
    fs_wrapper::remove_file("foo");
    assert_eq!(fs_wrapper::read_dir(cwd()).count(), 1);

    rustc().input("foo.rs").emit("llvm-bc").output("foo").run();
    fs_wrapper::remove_file("foo");
    rustc().input("foo.rs").emit("llvm-bc=foo").run();
    fs_wrapper::remove_file("foo");
    rustc().input("foo.rs").arg("--emit=llvm-bc=foo").run();
    fs_wrapper::remove_file("foo");
    assert_eq!(fs_wrapper::read_dir(cwd()).count(), 1);

    rustc().input("foo.rs").emit("llvm-ir").output("foo").run();
    fs_wrapper::remove_file("foo");
    rustc().input("foo.rs").emit("llvm-ir=foo").run();
    fs_wrapper::remove_file("foo");
    rustc().input("foo.rs").arg("--emit=llvm-ir=foo").run();
    fs_wrapper::remove_file("foo");
    assert_eq!(fs_wrapper::read_dir(cwd()).count(), 1);

    rustc().input("foo.rs").emit("obj").output("foo").run();
    fs_wrapper::remove_file("foo");
    rustc().input("foo.rs").emit("obj=foo").run();
    fs_wrapper::remove_file("foo");
    rustc().input("foo.rs").arg("--emit=obj=foo").run();
    fs_wrapper::remove_file("foo");
    assert_eq!(fs_wrapper::read_dir(cwd()).count(), 1);

    let bin_foo = bin_name("foo");
    rustc().input("foo.rs").emit("link").output(&bin_foo).run();
    fs_wrapper::remove_file(&bin_foo);
    rustc().input("foo.rs").emit(&format!("link={bin_foo}")).run();
    fs_wrapper::remove_file(&bin_foo);
    rustc().input("foo.rs").arg(&format!("--emit=link={bin_foo}")).run();
    fs_wrapper::remove_file(&bin_foo);
    std::fs::remove_file("foo.pdb").unwrap_or_default();
    assert_eq!(fs_wrapper::read_dir(cwd()).count(), 1);

    rustc().input("foo.rs").crate_type("rlib").output("foo").run();
    fs_wrapper::remove_file("foo");
    rustc().input("foo.rs").crate_type("rlib").emit("link=foo").run();
    fs_wrapper::remove_file("foo");
    rustc().input("foo.rs").crate_type("rlib").arg("--emit=link=foo").run();
    fs_wrapper::remove_file("foo");
    assert_eq!(fs_wrapper::read_dir(cwd()).count(), 1);

    rustc().input("foo.rs").crate_type("dylib").output(&bin_foo).run();
    fs_wrapper::remove_file(&bin_foo);
    rustc().input("foo.rs").crate_type("dylib").emit(&format!("link={bin_foo}")).run();
    fs_wrapper::remove_file(&bin_foo);
    rustc().input("foo.rs").crate_type("dylib").arg(&format!("--emit=link={bin_foo}")).run();
    fs_wrapper::remove_file(&bin_foo);
    remove_artifacts();
    assert_eq!(fs_wrapper::read_dir(cwd()).count(), 1);

    rustc().input("foo.rs").crate_type("staticlib").emit("link").output("foo").run();
    fs_wrapper::remove_file("foo");
    rustc().input("foo.rs").crate_type("staticlib").emit("link=foo").run();
    fs_wrapper::remove_file("foo");
    rustc().input("foo.rs").crate_type("staticlib").arg("--emit=link=foo").run();
    fs_wrapper::remove_file("foo");
    assert_eq!(fs_wrapper::read_dir(cwd()).count(), 1);

    rustc().input("foo.rs").crate_type("bin").output(&bin_foo).run();
    fs_wrapper::remove_file(&bin_foo);
    rustc().input("foo.rs").crate_type("bin").emit(&format!("link={bin_foo}")).run();
    fs_wrapper::remove_file(&bin_foo);
    rustc().input("foo.rs").crate_type("bin").arg(&format!("--emit=link={bin_foo}")).run();
    fs_wrapper::remove_file(&bin_foo);
    std::fs::remove_file("foo.pdb").unwrap_or_default();
    assert_eq!(fs_wrapper::read_dir(cwd()).count(), 1);

    rustc().input("foo.rs").emit("llvm-ir=ir").emit("link").crate_type("rlib").run();
    fs_wrapper::remove_file("ir");
    fs_wrapper::remove_file(rust_lib_name("bar"));
    assert_eq!(fs_wrapper::read_dir(cwd()).count(), 1);

    rustc()
        .input("foo.rs")
        .emit("asm=asm")
        .emit("llvm-ir=ir")
        .emit("llvm-bc=bc")
        .emit("obj=obj")
        .emit("link=link")
        .crate_type("staticlib")
        .run();
    fs_wrapper::remove_file("asm");
    fs_wrapper::remove_file("ir");
    fs_wrapper::remove_file("bc");
    fs_wrapper::remove_file("obj");
    fs_wrapper::remove_file("link");
    assert_eq!(fs_wrapper::read_dir(cwd()).count(), 1);

    rustc()
        .input("foo.rs")
        .arg("--emit=asm=asm")
        .arg("--emit")
        .arg("llvm-ir=ir")
        .arg("--emit=llvm-bc=bc")
        .arg("--emit")
        .arg("obj=obj")
        .arg("--emit=link=link")
        .crate_type("staticlib")
        .run();
    fs_wrapper::remove_file("asm");
    fs_wrapper::remove_file("ir");
    fs_wrapper::remove_file("bc");
    fs_wrapper::remove_file("obj");
    fs_wrapper::remove_file("link");
    assert_eq!(fs_wrapper::read_dir(cwd()).count(), 1);

    rustc().input("foo.rs").emit("asm,llvm-ir,llvm-bc,obj,link").crate_type("staticlib").run();
    fs_wrapper::remove_file("bar.ll");
    fs_wrapper::remove_file("bar.s");
    fs_wrapper::remove_file("bar.o");
    fs_wrapper::remove_file(static_lib_name("bar"));
    fs_wrapper::rename("bar.bc", "foo.bc");
    // Don't check that no files except foo.rs remain - we left `foo.bc` for later
    // comparison.

    rustc().input("foo.rs").emit("llvm-bc,link").crate_type("rlib").run();
    assert_eq!(fs_wrapper::read("foo.bc"), fs_wrapper::read("bar.bc"));
    fs_wrapper::remove_file("bar.bc");
    fs_wrapper::remove_file("foo.bc");
    fs_wrapper::remove_file(rust_lib_name("bar"));
    assert_eq!(fs_wrapper::read_dir(cwd()).count(), 1);
}
