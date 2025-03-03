use run_make_support::rustc;

fn main() {
    rustc().input("libr.rs").run();
    rustc().input("app.rs").run_fail().assert_stderr_contains(
        "`extern dyn` annotation is only avaible for dynamic dependencies with stable interface",
    );
}
