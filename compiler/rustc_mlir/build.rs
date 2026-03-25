use std::env;
use std::error::Error;
use std::path::PathBuf;

use cargo_metadata::MetadataCommand;
use rustc_helper::llvm::Llvm;
use rustc_helper::triton::Triton;

fn main() -> Result<(), Box<dyn Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Get project directory and target directory using cargo_metadata
    // Use rustc_llvm directory since that's where llvm.toml and triton.toml are located
    let metadata = MetadataCommand::new().exec().unwrap();
    let root_dir: PathBuf = metadata.workspace_root.into();
    let target_dir: PathBuf = metadata.target_directory.into();

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let wrapper_dir = manifest_dir.join("mlir-wrapper");

    // Get LLVM and Triton configs using helper functions
    let llvm = Llvm::new(&root_dir, &target_dir);
    let triton = Triton::new(&root_dir, &target_dir);

    // Use LLVM install_dir for cmake config paths
    let llvm_build_dir = llvm.out_dir.join("build");
    let triton_build_dir = triton.out_dir.join("build");
    let mlir_dir = llvm_build_dir.join("lib/cmake/mlir");

    let mlir_wrapper_build_dir = root_dir.join("target/build/mlir-wrapper-build");

    // Ensure the mlir-wrapper build directory exists before configuring cmake
    if let Err(e) = std::fs::create_dir_all(&mlir_wrapper_build_dir) {
        panic!("Failed to create mlir-wrapper build directory {:?}: {}", mlir_wrapper_build_dir, e);
    }

    // Configure cmake build
    let mut config = cmake::Config::new(&wrapper_dir);

    config
        .generator("Ninja")
        .out_dir(mlir_wrapper_build_dir)
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("LLVM_DIR", llvm_build_dir.join("lib/cmake/llvm"))
        .define("MLIR_DIR", &mlir_dir)
        .define("TRITON_SOURCE_DIR", triton.source_dir())
        .define("TRITON_BUILD_DIR", triton_build_dir);

    let dst = config.build();

    println!(
        "cargo:rustc-env=LLVM_INCLUDE_DIRECTORY={}",
        llvm.install_dir.join("include").display()
    );

    println!(
        "cargo:rustc-env=TRITON_INCLUDE_DIRECTORY={}",
        triton.source_dir().join("include").display()
    );

    // Link the built library
    println!("cargo:rustc-link-search=native={}", dst.join("lib").display());
    println!("cargo:rustc-link-lib=static=mlir-wrapper");

    // Link MLIR libraries
    let mlir_lib_dir = llvm.out_dir.join("build/lib");
    println!("cargo:rustc-link-search=native={}", mlir_lib_dir.display());

    // Link LLVM support libraries
    println!("cargo:rustc-link-lib=LLVMSupport");
    println!("cargo:rustc-link-lib=LLVMCore");

    // Link Triton libraries
    println!("cargo:rustc-link-search=native={}", triton.link_dir().display());
    for lib in triton.link_libs() {
        println!("cargo:rustc-link-lib={}", lib);
    }

    // Triton's SPIR-V translation path depends on SPIRV-Tools (C++ + C API).
    // These symbols are referenced from libtriton.a (e.g. SPIRVTranslation.cpp.o),
    // so we must ensure the final link includes SPIRV-Tools libraries.
    //
    // On most Linux distros these come from the `spirv-tools` / `spirv-tools-dev` packages
    // and live on the default linker search path.
    println!("cargo:rustc-link-lib=SPIRV-Tools");
    println!("cargo:rustc-link-lib=SPIRV-Tools-opt");
    println!("cargo:rustc-link-lib=SPIRV-Tools-link");

    // Link C++ standard library
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }

    // Rerun if wrapper sources change
    println!("cargo:rerun-if-changed=mlir-wrapper/");
    // Note: llvm.toml and triton.toml are in rustc_llvm directory, tracked by rustc_llvm's build.rs

    // Generate bindings header path for use in lib.rs
    println!("cargo:include={}", wrapper_dir.display());
    println!("cargo:out_dir={}", out_dir.display());

    Ok(())
}
