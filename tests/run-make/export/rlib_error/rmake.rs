use run_make_support::rustc;

fn main() {
    rustc().input("libr.rs").arg("-Csymbol-mangling-version=v0").run();
    rustc().input("app.rs").arg("-Csymbol-mangling-version=v0").run_fail().assert_stderr_contains(
        "`extern dyn` annotation is only avaible for dynamic dependencies with stable interface",
    );
}
