// Make sure that for `aarch64-unknown-linux-pauthtest` compiler emits warning when static
// libraries are linked. Test both foreign module linked from #[link] directive and command line
// invocations.

//@ only-aarch64-unknown-linux-pauthtest
// ignore-tidy-linelength

use run_make_support::{cc, env_var, regex, run, rustc};

fn main() {
    let input = "helper";
    let input_name = format!("{input}.c");
    let lib_name = format!("{}{input}.{}", "lib", "a");
    // Build a static library
    cc().out_exe(&lib_name)
        .input(&input_name)
        .args(&["-target", "aarch64-unknown-linux-pauthtest", "-march=armv8.3-a+pauth", "-c"])
        .run();

    // Check against foreign module warning: #[link(name = "helper", kind = "static")]
    let stderr_foreign_module = rustc()
        .target("aarch64-unknown-linux-pauthtest")
        .input("main.rs")
        .linker(&env_var("CC"))
        .link_args(&env_var("CC_DEFAULT_FLAGS"))
        .arg("-L.")
        .run()
        .stderr_utf8();
    run("main");
    let re_foreign_moule = regex::Regex::new( r"(?s)warning: library `helper`.*linked statically.*aarch64-unknown-linux-pauthtest.*requires dynamic linking.*using dynamic linking instead").unwrap();
    assert!(re_foreign_moule.is_match(&stderr_foreign_module));

    // Check against command line warning: -lstatic=helper
    let stderr_command_line = rustc()
        .target("aarch64-unknown-linux-pauthtest")
        .input("main_cmd_line.rs")
        .linker(&env_var("CC"))
        .link_args(&env_var("CC_DEFAULT_FLAGS"))
        .arg("-L.")
        .arg("-lstatic=helper")
        .run()
        .stderr_utf8();
    run("main_cmd_line");
    let re_cmd_line = regex::Regex::new( r"(?s)warning: static linking of `helper`.*is not supported on.*aarch64-unknown-linux-pauthtest.*using dynamic linking instead").unwrap();
    assert!(re_cmd_line.is_match(&stderr_command_line));
}
