use std::{path::PathBuf, process::Command};
#[cfg(feature = "bindgen")]
use {bindgen, std::fs};
use std::env;

fn main() {
    let base_dir = env!("CARGO_MANIFEST_DIR");
    let base_dir = PathBuf::from(&base_dir);                                        
    let build_dir = base_dir.join("enzyme").join("build");
    if !std::path::Path::new(&build_dir).exists() {
        std::fs::create_dir(&build_dir).unwrap();
    }

    let rust_build_dir = base_dir.parent().unwrap()
        .parent().unwrap()
        .parent().unwrap();
    let llvm_dir = rust_build_dir.join("build")
        .join("x86_64-unknown-linux-gnu").join("llvm");
    let llvm_lib_dir = llvm_dir.join("lib").join("cmake").join("llvm");
    let llvm_external_lib = "-DENZYME_EXTERNAL_SHARED_LIB=ON";
    let build_type = "-DCMAKE_BUILD_TYPE=Release";

    let mut cmake = Command::new("cmake");
    let mut build = Command::new("cmake");
    cmake
        .args(&[
              "-G",
              "Ninja",
              "..",
              build_type,
              &llvm_external_lib,
              &("-DLLVM_DIR=".to_owned() + &llvm_lib_dir.to_string_lossy()),
        ])
        .current_dir(&build_dir.to_str().unwrap());
    build.args(&["--build","."])
        .current_dir(&build_dir.to_str().unwrap());
    run_and_printerror(&mut cmake);
    run_and_printerror(&mut build);

    #[cfg(feature = "bindgen")]
    generate_bindings(base_dir, llvm_dir);
}

fn run_and_printerror(command: &mut Command) {
    println!("Running: `{:?}`", command);
    match command.status() {
        Ok(status) => {
            if !status.success() {
                panic!("Failed: `{:?}` ({})", command, status);
            }
        }
        Err(error) => {
            panic!("Failed: `{:?}` ({})", command, error);
        }
    }
}

#[cfg(feature = "bindgen")]
fn generate_bindings(base_dir: PathBuf, llvm_dir: PathBuf) {
    let llvm_headers = llvm_dir.join("include");
    let out_file = base_dir.join("enzyme.rs");
    let capi_header = base_dir.join("enzyme").join("Enzyme").join("CApi.h");
    print!("{:?}", capi_header.display());

    let content: String = fs::read_to_string(capi_header.clone()).unwrap();

    let bindings = bindgen::Builder::default()
        .header_contents("CApi.hpp", &content) // read it as .hpp so bindgen can ignore the class successfully
        .clang_args(&[format!(
                "-I{}",
                llvm_headers.display()
                )])
        .allowlist_type("CConcreteType")
        .rustified_enum("CConcreteType")
        .allowlist_type("CDerivativeMode")
        .rustified_enum("CDerivativeMode")
        .allowlist_type("CDIFFE_TYPE")
        .rustified_enum("CDIFFE_TYPE")
        .allowlist_type("CTypeTreeRef")
        .allowlist_type("EnzymeTypeAnalysisRef")
        .allowlist_function("EnzymeNewTypeTree")
        .allowlist_function("EnzymeFreeTypeTree")
        // Next two are for debugging / printning type information
        .allowlist_function("EnzymeSetCLBool")
        .allowlist_function("EnzymeSetCLInteger")
        .allowlist_function("CreateTypeAnalysis")
        .allowlist_function("ClearTypeAnalysis")
        .allowlist_function("FreeTypeAnalysis")
        .allowlist_function("CreateEnzymeLogic")
        .allowlist_function("ClearEnzymeLogic")
        .allowlist_function("FreeEnzymeLogic")
        .allowlist_function("EnzymeCreateForwardDiff")
        .allowlist_function("EnzymeCreatePrimalAndGradient")
        .allowlist_function("EnzymeCreateAugmentedPrimal")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Finish the builder and generate the bindings.
        .generate();

    let bindings = match bindings {
        Ok(v) => v,
        Err(_) => {
            panic!(
                "Unable to generate bindings from {}.",
                capi_header.display()
                )
        }
    };


    if out_file.exists() {
        fs::remove_file(out_file.clone()).unwrap();
    }
    let result = bindings.write_to_file(out_file.clone());

    if let Err(_) = result {
        panic!("Couldn't write bindings to {}.",
               out_file.display()
              )
    }
}
