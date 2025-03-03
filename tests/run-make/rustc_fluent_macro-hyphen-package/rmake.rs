use run_make_support::external_deps::cargo::cargo;

// test to make sure that `rustc_fluent_macro` correctly handles
// packages that have hyphens in their package name.

fn main() {
    cargo().arg("build").arg("--manifest-path=./Cargo.toml").run();
}
