extern crate build_helper;

use std::process::Command;
use build_helper::{Config, build_static_lib, Run, LLVMTools};

fn main() {
    build_rustllvm();
    generate_llvmdeps();
}

fn build_rustllvm() {
    let cfg = Config::new();
    let src_dir = cfg.src_dir().join("rustllvm");
    let src_files = vec!["ExecutionEngineWrapper.cpp",
                         "PassWrapper.cpp", "RustWrapper.cpp"];
    build_static_lib(&cfg)
        .set_src_dir(&src_dir)
        .set_build_dir(&cfg.out_dir())
        .files(&src_files)
        .with_llvm()
        .compile("rustllvm");
}

fn generate_llvmdeps() {
    let cfg = Config::new();
    let script = cfg.src_dir().join("etc").join("mklldeps.py");
    let dest = cfg.out_dir().join("llvmdeps.rs");
    let llvm_tools = LLVMTools::new(&cfg);
    let llvm_components = "x86 arm aarch64 mips powerpc ipo bitreader \
                           bitwriter linker asmparser mcjit interpreter \
                           instrumentation";
    Command::new("python")
        .arg(&script)
        .arg(&dest)
        .arg(llvm_components)
        .arg(if cfg!(feature = "static-libstdcpp") { "1" } else { "" })
        .arg(&llvm_tools.path_to_llvm_config())
        .run();
    println!("cargo:rustc-link-search={}/lib",
             llvm_tools.path_to_llvm_libs().display());
}
