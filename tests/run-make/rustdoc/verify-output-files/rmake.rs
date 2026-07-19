//@ needs-target-std

use std::path::{Path, PathBuf};

use run_make_support::{assert_dirs_are_equal, rfs, rustdoc};

#[derive(PartialEq)]
enum JsonOutput {
    Yes,
    No,
}

fn generate_docs(out_dir: &Path, json_output: JsonOutput) {
    let mut rustdoc = rustdoc();
    rustdoc.input("src/lib.rs").crate_name("foobar").crate_type("lib").out_dir(&out_dir);
    if json_output == JsonOutput::Yes {
        rustdoc.arg("-Zunstable-options").output_format("json");
    }
    rustdoc.run();
}

fn main() {
    let out_dir = PathBuf::from("rustdoc");
    let tmp_out_dir = PathBuf::from("tmp-rustdoc");

    // Generate HTML docs.
    generate_docs(&out_dir, JsonOutput::No);

    // Copy first output for to check if it's exactly same after second compilation.
    rfs::copy_dir_all(&out_dir, &tmp_out_dir);

    // Generate html docs once again on same output.
    generate_docs(&out_dir, JsonOutput::No);

    // Generate json doc on the same output.
    generate_docs(&out_dir, JsonOutput::Yes);

    // Check if expected json file is generated.
    assert!(out_dir.join("foobar.json").is_file());

    // Copy first json output to check if it's exactly same after second compilation.
    rfs::copy(out_dir.join("foobar.json"), tmp_out_dir.join("foobar.json"));

    // Generate json doc on the same output.
    generate_docs(&out_dir, JsonOutput::Yes);

    // Check if all docs(including both json and html formats) are still the same after multiple
    // compilations.
    assert_dirs_are_equal(&out_dir, &tmp_out_dir);
}
