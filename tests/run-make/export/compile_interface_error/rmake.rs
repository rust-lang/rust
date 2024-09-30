use run_make_support::rustc;

fn main() {
    // Do not produce the interface, use the broken one.
    rustc().input("libr.rs").arg("-Csymbol-mangling-version=v0").run();
    rustc()
        .input("app.rs")
        .arg("-Csymbol-mangling-version=v0")
        .run_fail()
        .assert_stderr_contains("couldn't compile interface");
}
