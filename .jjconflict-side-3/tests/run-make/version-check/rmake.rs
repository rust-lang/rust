use run_make_support::bare_rustc;

fn main() {
    let signalled_version = "Ceci n'est pas une rustc";
    let rustc_out = bare_rustc()
        .env("RUSTC_OVERRIDE_VERSION_STRING", signalled_version)
        .arg("--version")
        .run()
        .stdout_utf8();

    let version = rustc_out.strip_prefix("rustc ").unwrap().trim_end();
    assert_eq!(version, signalled_version);
}
