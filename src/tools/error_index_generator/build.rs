use std::path::PathBuf;
use std::{env, fs};
use walkdir::WalkDir;

fn main() {
    // The src directory (we are in src/tools/error_index_generator)
    // Note that we could skip one of the .. but this ensures we at least loosely find the right
    // directory.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let error_codes_path = "../../../compiler/rustc_error_codes/src/error_codes.rs";

    println!("cargo:rerun-if-changed={}", error_codes_path);
    let file = fs::read_to_string(error_codes_path)
        .unwrap()
        .replace(": include_str!(\"./error_codes/", ": include_str!(\"./");
    let contents = format!("(|| {{\n{}\n}})()", file);
    fs::write(&out_dir.join("all_error_codes.rs"), &contents).unwrap();

    // We copy the md files as well to the target directory.
    for entry in WalkDir::new("../../../compiler/rustc_error_codes/src/error_codes") {
        let entry = entry.unwrap();
        match entry.path().extension() {
            Some(s) if s == "md" => {}
            _ => continue,
        }
        println!("cargo:rerun-if-changed={}", entry.path().to_str().unwrap());
        let md_content = fs::read_to_string(entry.path()).unwrap();
        fs::write(&out_dir.join(entry.file_name()), &md_content).unwrap();
    }
}
