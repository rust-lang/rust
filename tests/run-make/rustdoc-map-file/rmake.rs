use run_make_support::{rustdoc, tmp_dir};
use std::process::Command;

fn main() {
    let out_dir = tmp_dir().join("out");
    rustdoc()
        .input("foo.rs")
        .arg("-Zunstable-options")
        .arg("--generate-redirect-map")
        .output(&out_dir)
        .run();
    // FIXME (GuillaumeGomez): Port the python script to Rust as well.
    let python = std::env::var("PYTHON").unwrap_or("python".into());
    assert!(Command::new(python).arg("validate_json.py").arg(&out_dir).status().unwrap().success());
}
