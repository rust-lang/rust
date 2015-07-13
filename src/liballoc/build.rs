extern crate build_helper;

use std::process::Command;
use std::fs::copy;
use build_helper::{Config, Run, GccishToolchain};

fn main() {
    build_jemalloc();
    // FIXME: It seems that when building tests, Cargo missed one of the
    // link search libraries from the transitive dependencies. In this
    // case, both libcollections and libstd depend on liballoc, the latter
    // of which also depends on libcollections.
    //
    // Because liballoc is placed into <target>/<profile> when building tests,
    // which is not passed to rustc as a dependency search directory, rustc
    // complains about 'possible newer version of crate' because it is
    // looking at the liballoc from the sysroot.
    //
    // We workaround this by manually passing this directory to rustc.
    let root_dir = Config::new().out_dir()
        .join("..").join("..").join("..");
    println!("cargo:rustc-link-search=dependency={}", root_dir.display());
}

fn build_jemalloc() {
    let cfg = Config::new();

    // We ignore jemalloc on windows for the time-being, as `bash` is not
    // universally available on Windows.
    let build_jemalloc =
        cfg!(feature = "jemalloc") || !cfg.target().is_windows();
    if !build_jemalloc {
        return
    }

    println!("cargo:rustc-cfg=jemalloc");

    let src_dir = cfg.src_dir().join("jemalloc");
    let build_dir = cfg.out_dir();
    let target = cfg.target();

    let mut cmd = Command::new("sh");
    cmd.arg(&src_dir.join("configure"));
    cmd.current_dir(&build_dir);
    if target.is_mingw() {
        // This isn't necessarily a desired option, but it's harmless and
        // works around what appears to be a mingw-w64 bug.
        //
        // https://sourceforge.net/p/mingw-w64/bugs/395/
        cmd.arg("--enable-lazy-lock");
    } else if target.is_ios() || target.is_android() {
        cmd.arg("--disable-tls");
    }

    if cfg!(feature = "debug-jemalloc") {
        cmd.arg("--enable-debug");
    }

    // Turn off broken quarantine (see jemalloc/jemalloc#161)
    cmd.arg("--disable-fill");

    match &cfg.profile()[..] {
        "bench" | "release" => {}
        _ => { cmd.arg("--enable-debug"); }
    }

    cmd.arg("--with-jemalloc-prefix=je_");
    cmd.arg(format!("--host={}", target));

    let gcc = GccishToolchain::new(target);
    let cflags = gcc.cflags().connect(" ");
    cmd.arg(format!("CC={}", gcc.cc_cmd));
    cmd.arg(format!("AR={}", gcc.ar_cmd));
    cmd.arg(format!("RANLIB={} s", gcc.ar_cmd));
    cmd.arg(format!("EXTRA_CFLAGS=-g1 {}", cflags));
    cmd.run();

    Command::new("make").current_dir(&build_dir).arg("build_lib_static").run();

    let _ = copy(build_dir.join("lib").join("libjemalloc_pic.a"),
                 build_dir.join("libjemalloc.a")).unwrap();
    println!("cargo:rustc-link-search=native={}", build_dir.display());
}
