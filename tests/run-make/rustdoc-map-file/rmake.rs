use run_make_support::{python_command, rustdoc, tmp_dir};

fn main() {
    let out_dir = tmp_dir().join("out");
    rustdoc()
        .input("foo.rs")
        .arg("-Zunstable-options")
        .arg("--generate-redirect-map")
        .output(&out_dir)
        .run();
    // FIXME (GuillaumeGomez): Port the python script to Rust as well.
    assert!(python_command().arg("validate_json.py").arg(&out_dir).status().unwrap().success());
}
