extern crate gcc;
extern crate tempdir;

use support::Test;

mod support;

#[test]
fn gnu_smoke() {
    let test = Test::gnu();
    test.gcc()
        .file("foo.c").compile("libfoo.a");

    test.cmd(0).must_have("-O2")
               .must_have("foo.c")
               .must_not_have("-g")
               .must_have("-c")
               .must_have("-ffunction-sections")
               .must_have("-fdata-sections");
    test.cmd(1).must_have(test.td.path().join("foo.o"));
}

#[test]
fn gnu_opt_level_1() {
    let test = Test::gnu();
    test.gcc()
        .opt_level(1)
        .file("foo.c").compile("libfoo.a");

    test.cmd(0).must_have("-O1")
               .must_not_have("-O2");
}

#[test]
fn gnu_opt_level_s() {
    let test = Test::gnu();
    test.gcc()
        .opt_level_str("s")
        .file("foo.c").compile("libfoo.a");

    test.cmd(0).must_have("-Os")
               .must_not_have("-O1")
               .must_not_have("-O2")
               .must_not_have("-O3")
               .must_not_have("-Oz");
}

#[test]
fn gnu_debug() {
    let test = Test::gnu();
    test.gcc()
        .debug(true)
        .file("foo.c").compile("libfoo.a");
    test.cmd(0).must_have("-g");
}

#[test]
fn gnu_x86_64() {
    for vendor in &["unknown-linux-gnu", "apple-darwin"] {
        let target = format!("x86_64-{}", vendor);
        let test = Test::gnu();
        test.gcc()
            .target(&target)
            .host(&target)
            .file("foo.c").compile("libfoo.a");

        test.cmd(0).must_have("-fPIC")
                   .must_have("-m64");
    }
}

#[test]
fn gnu_x86_64_no_pic() {
    for vendor in &["unknown-linux-gnu", "apple-darwin"] {
        let target = format!("x86_64-{}", vendor);
        let test = Test::gnu();
        test.gcc()
            .pic(false)
            .target(&target)
            .host(&target)
            .file("foo.c").compile("libfoo.a");

        test.cmd(0).must_not_have("-fPIC");
    }
}

#[test]
fn gnu_i686() {
    for vendor in &["unknown-linux-gnu", "apple-darwin"] {
        let target = format!("i686-{}", vendor);
        let test = Test::gnu();
        test.gcc()
            .target(&target)
            .host(&target)
            .file("foo.c").compile("libfoo.a");

        test.cmd(0).must_not_have("-fPIC")
                   .must_have("-m32");
    }
}

#[test]
fn gnu_i686_pic() {
    for vendor in &["unknown-linux-gnu", "apple-darwin"] {
        let target = format!("i686-{}", vendor);
        let test = Test::gnu();
        test.gcc()
            .pic(true)
            .target(&target)
            .host(&target)
            .file("foo.c").compile("libfoo.a");

        test.cmd(0).must_have("-fPIC");
    }
}

#[test]
fn gnu_set_stdlib() {
    let test = Test::gnu();
    test.gcc()
        .cpp_set_stdlib(Some("foo"))
        .file("foo.c").compile("libfoo.a");

    test.cmd(0).must_not_have("-stdlib=foo");
}

#[test]
fn gnu_include() {
    let test = Test::gnu();
    test.gcc()
        .include("foo/bar")
        .file("foo.c").compile("libfoo.a");

    test.cmd(0).must_have("-I").must_have("foo/bar");
}

#[test]
fn gnu_define() {
    let test = Test::gnu();
    test.gcc()
        .define("FOO", Some("bar"))
        .define("BAR", None)
        .file("foo.c").compile("libfoo.a");

    test.cmd(0).must_have("-DFOO=bar").must_have("-DBAR");
}

#[test]
fn gnu_compile_assembly() {
    let test = Test::gnu();
    test.gcc()
        .file("foo.S").compile("libfoo.a");
    test.cmd(0).must_have("foo.S");
}

#[test]
fn msvc_smoke() {
    let test = Test::msvc();
    test.gcc()
        .file("foo.c").compile("libfoo.a");

    test.cmd(0).must_have("/O2")
               .must_have("foo.c")
               .must_not_have("/Z7")
               .must_have("/c");
    test.cmd(1).must_have(test.td.path().join("foo.o"));
}

#[test]
fn msvc_opt_level_0() {
    let test = Test::msvc();
    test.gcc()
        .opt_level(0)
        .file("foo.c").compile("libfoo.a");

    test.cmd(0).must_not_have("/O2");
}

#[test]
fn msvc_debug() {
    let test = Test::msvc();
    test.gcc()
        .debug(true)
        .file("foo.c").compile("libfoo.a");
    test.cmd(0).must_have("/Z7");
}

#[test]
fn msvc_include() {
    let test = Test::msvc();
    test.gcc()
        .include("foo/bar")
        .file("foo.c").compile("libfoo.a");

    test.cmd(0).must_have("/I").must_have("foo/bar");
}

#[test]
fn msvc_define() {
    let test = Test::msvc();
    test.gcc()
        .define("FOO", Some("bar"))
        .define("BAR", None)
        .file("foo.c").compile("libfoo.a");

    test.cmd(0).must_have("/DFOO=bar").must_have("/DBAR");
}
