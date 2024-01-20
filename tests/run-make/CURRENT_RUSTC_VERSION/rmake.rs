// ignore-tidy-linelength

extern crate run_make_support;

use run_make_support::{aux_build, rustc};

fn main() {
    aux_build("--emit=metadata stable.rs");
    let output = rustc(&format!(
        "--emit=metadata --extern stable={}/libstable.rmeta main.rs",
        env!("TMPDIR")
    ));

    let stderr = String::from_utf8_lossy(&output.stderr);
    let version = include_str!(concat!(env!("S"), "/src/version"));
    let expected_string = format!("stable since {}", version.trim());
    assert!(stderr.contains(&expected_string));
}
