use crate::{
    code::utils::{
        get_bindings_path, get_capi_path, get_local_enzyme_repo_path, get_local_rust_repo_path,
    },
    Cli,
};

use super::utils;
use bindgen;
use std::fs;

/// This function can be used to generate Rust wrappers around Enzyme's [C API](https://github.com/wsmoses/Enzyme/blob/main/enzyme/Enzyme/CApi.h).
//pub fn generate_bindings_with(capi_header: PathBuf, out_file: PathBuf) -> Result<(), String> {
pub fn generate_bindings(args: Cli) -> Result<(), String> {
    let enzyme_repo = get_local_enzyme_repo_path(args.enzyme);
    let rust_repo = get_local_rust_repo_path(args.rust);

    let capi_header = get_capi_path(enzyme_repo);
    dbg!(&capi_header);

    // tell cargo to re-run the builder if the header has changed
    println!("cargo:rerun-if-changed={}", capi_header.display());
    let content: String = fs::read_to_string(capi_header.clone()).unwrap();

    let bindings = bindgen::Builder::default()
        .header_contents("CApi.hpp", &content) // read it as .hpp so bindgen can ignore the class successfully
        .clang_args(&[format!(
            "-I{}",
            utils::get_llvm_header_path(rust_repo).display()
        )])
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Finish the builder and generate the bindings.
        .generate();

    let bindings = match bindings {
        Ok(v) => v,
        Err(_) => {
            return Err(format!(
                "Unable to generate bindings from {}.",
                capi_header.display()
            ))
        }
    };

    let out_file = get_bindings_path();

    if out_file.exists() {
        fs::remove_file(out_file.clone()).unwrap();
    }
    let result = bindings.write_to_file(out_file.clone());

    match result {
        Ok(_) => Ok(()),
        Err(_) => Err(format!(
            "Couldn't write bindings to {}.",
            out_file.display()
        )),
    }
}
